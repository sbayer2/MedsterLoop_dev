from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from medster.agent import Agent
from medster.utils.intro import print_intro
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory


def main():
    print_intro()
    agent = Agent()

    # Create a prompt session with history
    session = PromptSession(history=InMemoryHistory())

    while True:
        try:
            # Prompt the user for input
            query = session.prompt("medster>> ")
            if query.lower() in ["exit", "quit"]:
                print("Session ended. Goodbye!")
                break
            if query:
                # Run the clinical analysis agent
                agent.run(query)
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended. Goodbye!")
            break


if __name__ == "__main__":
    main()
