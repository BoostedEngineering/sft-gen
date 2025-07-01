from sft_gen.nlp_resources.groupings import generate_groupings

# Example discussions about different stocks
discussions = """
Apple's stock price increased by 5% today. Apple is launching a new iPhone next month.
Tesla announced a new model that will be released next year.
Tesla's stock price dropped after the announcement.
Microsoft's quarterly earnings exceeded expectations.
Microsoft is planning to acquire a gaming company.
"""

for group in generate_groupings(discussions):
    print(group)
