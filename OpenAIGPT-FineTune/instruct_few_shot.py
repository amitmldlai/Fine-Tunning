few_shot = """This system functions as a document classifier, determining the appropriate category for a given text input. 
It analyzes the content and structure of the text to identify its classification, ensuring precise and contextually relevant categorization.

The system will provide a single-word category label based on the document's type. 

Here are examples of classifying different documents:

Text: 2023 W-2 Wage and Tax Statement
Employee's social security number 123-45-XXXX
Employer identification number 12-3456789
Employee name: John Smith
Wages, tips, other compensation: $75,000.00
Social security wages: $75,000.00
Medicare wages and tips: $75,000.00
Federal income tax withheld: $15,750.00

Category: W2
---------------------------------
Text: UTILITY BILL Service Provider: City Electric Co.
Account Number: 987654321
Billing Period: 01/01/2024 - 01/31/2024
Total Amount Due: $145.75
Due Date: 02/10/2024
Previous Balance: $140.50
Payments: -$140.50
Current Charges: $145.75

CATEGORY: UTILITY_BILL
"""