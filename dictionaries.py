# Add your code here

medical_costs={}

medical_costs['Marina']=6607.0
medical_costs['Vinay']=3225.0

#medical_costs= {k:v for k, v in (('Connie', 8886.0), ('Isaac', 16444.0), ('Valentina', 6420.0))}

medical_costs.update({'Connie': 8886.0, 'Isaac': 16444.0, 'Valentina': 6420.0})

medical_costs['Vinay']=3325.0

print(medical_costs)

total_costs= 0

for value in medical_costs.values():
  total_costs+=value

average_cost=total_costs/len(medical_costs.values())

print('Average Insurance Cost : '+ str(average_cost))

names=['Marina','Vinay','Connie','Isaac','Valentina']
ages=[27,24,43,35,52]

zipped_ages=zip(names,ages)

names_to_ages={k:v for k, v in zipped_ages}
print(names_to_ages)

marina_age=names_to_ages.get('Marina')

print("Marina's age is " + str(marina_age))

medical_records={}
medical_records['Marina']={"Age": 27, "Sex": "Female", "BMI": 31.1, "Children": 2, "Smoker": "Non-smoker", "Insurance_cost": 6607.0}

medical_records['Vinay']={"Age": 24, "Sex": "Male", "BMI": 26.9, "Children": 0, "Smoker": "Non-smoker", "Insurance_cost": 3225.0}

medical_records['Connie']={"Age": 43, "Sex": "Female", "BMI": 25.3, "Children": 3, "Smoker": "Non-smoker", "Insurance_cost": 8886.0}

medical_records['Isaac']={"Age": 35, "Sex": "Male", "BMI": 20.6, "Children": 4, "Smoker": "Smoker", "Insurance_cost": 16444.0}

medical_records['Valentina']={"Age": 52, "Sex": "Female", "BMI": 18.7, "Children": 1, "Smoker": "Non-smoker", "Insurance_cost": 6420.0}

print(medical_records)

print("Connie's insurance cost is " + str(medical_records['Connie']['Insurance_cost']) + " dollars")

medical_records.pop('Vinay')

for med in medical_records.keys():
  name= med
  age= medical_records[name]['Age']
  sex= medical_records[name]['Sex']
  smoker= medical_records[name]['Smoker']
  bmi=medical_records[name]['BMI']
  insurance=medical_records[name]["Insurance_cost"]

  print(name + " is a " + str(age) + " year old " + sex +" "+ smoker + " with a BMI of " + str(bmi) + " and insurance cost of " + str(insurance))