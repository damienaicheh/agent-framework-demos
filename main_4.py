import logging
import os

from agent_framework.azure import AzureAIClient
from agent_framework_devui import serve
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
                        Always provide the time required to fix the issue by using the TimePerIssueTools.
                    """,
        name="IssueAnalyzerAgent",
        default_options={"response_format": IssueAnalyzer},
        tools=[timePerIssueTools.calculate_time_based_on_complexity],
    )

    serve(entities=[issue_analyzer_agent], port=8090, auto_open=True)


if __name__ == "__main__":
    main()
