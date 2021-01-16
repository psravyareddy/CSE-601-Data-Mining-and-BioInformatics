## Project requirements

- Python - 3.7.0
- Pandas - 1.1.0
- Numpy - 1.19.1

## Running the project on the command line

Open the command prompt from your project folder and run the following command.

python data_mining_project_1_part_2.py --filepath <i><path_to_file></i> --support <i><support_value></i> --confidence <i><confidence_value></i>

Example: python data_mining_project_1_part_2.py --filepath association-rule-test-data.txt --support 50 --confidence 70

## Generating association rules based on template format
Type in a query as per the given format. <br>
All queries are handled in the code for case sensitivity. <br>

When rule generation begins enter the <i>template number</i> to input queries or <i>0 to quit</i> the program. <br>

Next enter all the template combinations as '|' separated query. <br>
Enter all the itemsets as a comma-separated list.

### Template 1
RULE|ANY|G59_UP <br>
BODY|1|G59_UP,G10_Down

### Template 2
RULE|3 <br>
BODY|1

### Template 3
1or1|HEAD|ANY|G10_Down|BODY|1|G59_UP <br>
1or2|HEAD|ANY|G10_Down|BODY|2 <br>
2and2|HEAD|1|BODY|2

