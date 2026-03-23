import logging
import os

from agent_framework import MCPStreamableHTTPTool
from agent_framework.azure import AzureAIClient
from agent_framework.orchestrations import GroupChatBuilder
from agent_framework_devui import serve
from agent_framework_orchestrations import SequentialBuilder
from azure.ai.projects.models import MCPTool
from azure.identity.aio import AzureCliCredential
from dotenv import load_dotenv

from models.issue_analyzer import IssueAnalyzer
from tools.time_per_issue_tools import TimePerIssueTools

load_dotenv()


def main():
    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    credential = AzureCliCredential()
    settings = {
        "project_endpoint": os.environ["AZURE_AI_PROJECT_ENDPOINT"],
        "model_deployment_name": os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"],
        "credential": credential,
    }

    timePerIssueTools = TimePerIssueTools()

    issue_analyzer_agent = AzureAIClient(**settings).as_agent(
        instructions="""
                        You are analyzing issues. 
                        If the ask is a feature request the complexity should be 'NA'.
                        If the issue is a bug, analyze the stack trace and provide the likely cause and complexity level.

                        CRITICAL: You MUST use the provided tools for ALL calculations:
                        1. First determine the complexity level
                        2. Use the available tools to calculate time and cost estimates based on that complexity
                        3. Never provide estimates without using the tools first

                        Your response should contain only values obtained from the tool calls.
                    """,
        name="IssueAnalyzerAgent",
        default_options={"response_format": IssueAnalyzer},
        tools=[timePerIssueTools.calculate_time_based_on_complexity],
    )

    github_client_agent = AzureAIClient(**settings)
    file_search_tool = github_client_agent.get_file_search_tool(
        vector_store_ids=[os.environ["VECTOR_STORE_ID"]]
    )
    github_agent = github_client_agent.as_agent(
        name="GitHubAgent",
        instructions=f"""
            You are a helpful assistant that can create GitHub issues following Contoso's guidelines.
            You work on this repository: {os.environ["GITHUB_REPOSITORY"]}
            
            CRITICAL WORKFLOW:
            1. ALWAYS use the File Search tool FIRST to search for "github issues guidelines" or "issue template" to find the proper formatting and structure
            2. Follow the Contoso GitHub Issues Guidelines found in the vector store
            3. Use the retrieved guidelines to format the issue properly with correct structure, labels, and format
            4. Then use the GitHub MCP tool to create the issue with the properly formatted content
            
            IMPORTANT: You MUST search for guidelines BEFORE creating any issue to ensure compliance with company standards.
        """,
        tools=[
            file_search_tool,
            MCPTool(
                server_label="GitHub",
                server_url="https://api.githubcopilot.com/mcp",
                require_approval="never",
                project_connection_id="GitHub",
            )
        ],
    )

    group_workflow = GroupChatBuilder(
        participants=[issue_analyzer_agent, github_agent],
        intermediate_outputs=True,
        orchestrator_agent=AzureAIClient(**settings).as_agent(
            name="IssueCreatorOrchestrator",
            instructions="""
                You are a workflow manager that coordinates issue creation.
                Decide which participant should speak next.

                Output rules are mandatory:
                - Return ONLY one raw JSON object.
                - Do NOT wrap JSON in markdown fences.
                - Do NOT add extra text before or after JSON.
                - Use exactly these keys: terminate, reason, next_speaker, final_message.
                - If terminate is false, next_speaker must be one of: IssueAnalyzerAgent, GitHubAgent.

                Workflow policy:
                1. Ask IssueAnalyzerAgent first to classify and estimate complexity.
                2. Then ask GitHubAgent to create the issue.
                3. Terminate once GitHubAgent confirms completion.
            """,
            default_options={"temperature": 0},
        ),
    ).build()

    ms_learn_mcp_tool = MCPStreamableHTTPTool(
        name="Microsoft Learn MCP",
        url="https://learn.microsoft.com/api/mcp",
    )

    ms_learn_agent = AzureAIClient(**settings).as_agent(
        name="DocsAgent",
        instructions="""
            You are a Microsoft documentation assistant.
            Mandatory rules:
            1. You must call the Microsoft Learn MCP tool before answering any user question.
            2. You are not allowed to answer from internal knowledge alone.
            3. Your final answer must be grounded only in Microsoft Learn MCP results.
            4. If no relevant result is found, explicitly say the information was not found in Microsoft Learn.
            5. If the tool is unavailable or fails, do not guess or fabricate; state that you cannot answer without Microsoft Learn MCP.
            6. Keep responses concise, accurate, and factual.
        """,
        tools=ms_learn_mcp_tool,
    )

    group_workflow_agent = group_workflow.as_agent(
        name="IssueCreationAgentGroup"
    )

    sequential_workflow = SequentialBuilder(
        participants=[ms_learn_agent, group_workflow_agent]).build()

    serve(entities=[issue_analyzer_agent, github_agent, group_workflow, ms_learn_agent, sequential_workflow],
          port=8090, auto_open=True)


if __name__ == "__main__":
    main()
