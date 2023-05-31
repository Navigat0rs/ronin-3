# import os
#
# def write_file_names(path, output_file):
#     file_names = os.listdir(path)
#     with open(output_file, 'w') as file:
#         for name in file_names:
#             file.write(name + '\n')
#
# # Usage example
# path = 'D:\\000_Mora\\FYP\\RONiN\\Ronin_DataSets\\train_all'  # Replace with the desired directory path
# output_file = 'D:\\000_Mora\\FYP\\RONiN\\Ronin_DataSets\\train_all\\train_list.txt'  # Replace with the desired output file name
#
# write_file_names(path, output_file)

#=====================================================================================================================================
import os

def write_file_names(path,seen_list):
    file_names = os.listdir(path)
    for name in file_names:
        seen_list.append(name.split("_")[0])
    return seen_list

# Usage example
seen_list=[]
path = 'D:\\000_Mora\\FYP\\RONiN\\Ronin_DataSets\\seen_subjects_test_set'  # Replace with the desired directory path

seen_list=write_file_names(path,seen_list)
seen_list=list(set(seen_list))
print(seen_list)