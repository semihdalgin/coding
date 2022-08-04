
medical_data = \
"""Marina Allison   ,27   ,   31.1 , 
#7010.0   ;Markus Valdez   ,   30, 
22.4,   #4050.0 ;Connie Ballard ,43 
,   25.3 , #12060.0 ;Darnell Weber   
,   35   , 20.6   , #7500.0;
Sylvie Charles   ,22, 22.1 
,#3022.0   ;   Vinay Padilla,24,   
26.9 ,#4620.0 ;Meredith Santiago, 51   , 
29.3 ,#16330.0;   Andre Mccarty, 
19,22.7 , #2900.0 ; 
Lorena Hodson ,65, 33.1 , #19370.0; 
Isaac Vu ,34, 24.8,   #7045.0"""

# Add your code here

print(medical_data)

updated_medical_data=medical_data.replace('#','$')

num_records=0

for i in updated_medical_data:
  if i == '$'  :
    num_records +=1

print('There are {} medical records in data'.format(num_records))

medical_data_split = updated_medical_data.split(';')


medical_records=[]

for i in medical_data_split:
  medical_records.append(i.split(','))

medical_records_clean=[]

for record in medical_records:
  record_clean=[]
  for item in record:
    record_clean= item.strip()
    medical_records_clean.append(record_clean)

print(medical_records_clean)

names=[]
ages=[]
bmis=[]
insurance_costs=[]

a=0
for record in medical_records_clean:
  if a==0:
    names.append(record)
    a+=1
  elif a==1:
    ages.append(record)
    a+=1
  elif a==2:
    bmis.append(record)
    a+=1
  elif a==3:
    insurance_costs.append(record)
    a=0

print('names= '+ str(names))
print('ages= ' + str(ages))
print('bmis= ' + str(bmis))
print('insurance= '+ str(insurance_costs))

total_bmi=0

for bm in bmis:
  total_bmi+= float(bm)

average_bmi= total_bmi/len(bmis)
print('Average BMI: ' + str(round(average_bmi,2)))
