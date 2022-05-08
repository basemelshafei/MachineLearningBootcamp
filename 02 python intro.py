my_age = 32
my_age = 33
my_age = my_age + 1

print(my_age)

restaurant_Bill = 36.17
service_charge = 0.125

print(restaurant_Bill*service_charge)
print(type(33))

prime_numbers_list = [3, 7, 61, 29, 199]
primeAndPeople_list = ["king arthur", 17, "basem"]

print(primeAndPeople_list[1])
print(type(prime_numbers_list))

numbers_array = ([3, 5, 8])
print(type(numbers_array))

import pandas as pd

data = pd.read_csv('lsd_math_score_data.csv')
only_LSD = data['LSD_ppm']
print(only_LSD)

data['test_subject'] = 'math'
data['test_subject'] = data['LSD_ppm'] * data['Avg_Math_Test_Score']
numbers = data[['LSD_ppm', 'Avg_Math_Test_Score']]
print(numbers)
print(data)
del data['test_subject']
print(data)






