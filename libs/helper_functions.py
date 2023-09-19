from csv import DictWriter


def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)


# Use the above created function to append a dictionary as a row in our csv file ‘students.csv’,
field_names = ['Id','Name','Course','City','Session']
row_dict = {'Id': 81,'Name': 'Sachin','Course':'Maths','City':'Mumbai','Session':'Evening'}
# Append a dict as a row in csv file
append_dict_as_row('students.csv', row_dict, field_names)