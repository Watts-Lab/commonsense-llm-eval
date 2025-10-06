from tqdm import tqdm
import pandas as pd

statements = pd.read_csv("data/raw_statement_corpus.csv")

# Capitalize the first letter and add a period
statements.statement = statements.statement.map(lambda s: s.capitalize() + ".")
statements.head()


def statement_to_prompt(statement):
    prompt = f"""Consider the statement, "{statement}" """
    return (
        prompt
        + """Do you agree with this statement? Start your answer with a "yes" or "no".""",
        prompt
        + """Do you think most people would agree with this statement? Start your answer with a "yes" or "no".""",
        prompt
        + """Do you think this statement is common sense? Start your answer with a "yes" or "no".""",
    )


statements[["q1", "q2", "q3"]] = ""

for i, statement in tqdm(enumerate(statements.statement)):
    statements.loc[i, ["q1", "q2", "q3"]] = statement_to_prompt(statement)

statements.to_csv("data/statements_and_prompts.csv", index=False)
