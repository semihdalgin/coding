import operator

# names of hurricanes
names = ['Cuba I', 'San Felipe II Okeechobee', 'Bahamas', 'Cuba II', 'CubaBrownsville', 'Tampico', 'Labor Day',
         'New England', 'Carol', 'Janet', 'Carla', 'Hattie', 'Beulah', 'Camille', 'Edith', 'Anita', 'David', 'Allen',
         'Gilbert', 'Hugo', 'Andrew', 'Mitch', 'Isabel', 'Ivan', 'Emily', 'Katrina', 'Rita', 'Wilma', 'Dean', 'Felix',
         'Matthew', 'Irma', 'Maria', 'Michael']

# months of hurricanes
months = ['October', 'September', 'September', 'November', 'August', 'September', 'September', 'September', 'September',
          'September', 'September', 'October', 'September', 'August', 'September', 'September', 'August', 'August',
          'September', 'September', 'August', 'October', 'September', 'September', 'July', 'August', 'September',
          'October', 'August', 'September', 'October', 'September', 'September', 'October']

# years of hurricanes
years = [1924, 1928, 1932, 1932, 1933, 1933, 1935, 1938, 1953, 1955, 1961, 1961, 1967, 1969, 1971, 1977, 1979, 1980,
         1988, 1989, 1992, 1998, 2003, 2004, 2005, 2005, 2005, 2005, 2007, 2007, 2016, 2017, 2017, 2018]

# maximum sustained winds (mph) of hurricanes
max_sustained_winds = [165, 160, 160, 175, 160, 160, 185, 160, 160, 175, 175, 160, 160, 175, 160, 175, 175, 190, 185,
                       160, 175, 180, 165, 165, 160, 175, 180, 185, 175, 175, 165, 180, 175, 160]

# areas affected by each hurricane
areas_affected = [['Central America', 'Mexico', 'Cuba', 'Florida', 'The Bahamas'],
                  ['Lesser Antilles', 'The Bahamas', 'United States East Coast', 'Atlantic Canada'],
                  ['The Bahamas', 'Northeastern United States'],
                  ['Lesser Antilles', 'Jamaica', 'Cayman Islands', 'Cuba', 'The Bahamas', 'Bermuda'],
                  ['The Bahamas', 'Cuba', 'Florida', 'Texas', 'Tamaulipas'], ['Jamaica', 'Yucatn Peninsula'],
                  ['The Bahamas', 'Florida', 'Georgia', 'The Carolinas', 'Virginia'],
                  ['Southeastern United States', 'Northeastern United States', 'Southwestern Quebec'],
                  ['Bermuda', 'New England', 'Atlantic Canada'], ['Lesser Antilles', 'Central America'],
                  ['Texas', 'Louisiana', 'Midwestern United States'], ['Central America'],
                  ['The Caribbean', 'Mexico', 'Texas'], ['Cuba', 'United States Gulf Coast'],
                  ['The Caribbean', 'Central America', 'Mexico', 'United States Gulf Coast'], ['Mexico'],
                  ['The Caribbean', 'United States East coast'],
                  ['The Caribbean', 'Yucatn Peninsula', 'Mexico', 'South Texas'],
                  ['Jamaica', 'Venezuela', 'Central America', 'Hispaniola', 'Mexico'],
                  ['The Caribbean', 'United States East Coast'], ['The Bahamas', 'Florida', 'United States Gulf Coast'],
                  ['Central America', 'Yucatn Peninsula', 'South Florida'],
                  ['Greater Antilles', 'Bahamas', 'Eastern United States', 'Ontario'],
                  ['The Caribbean', 'Venezuela', 'United States Gulf Coast'],
                  ['Windward Islands', 'Jamaica', 'Mexico', 'Texas'], ['Bahamas', 'United States Gulf Coast'],
                  ['Cuba', 'United States Gulf Coast'], ['Greater Antilles', 'Central America', 'Florida'],
                  ['The Caribbean', 'Central America'], ['Nicaragua', 'Honduras'],
                  ['Antilles', 'Venezuela', 'Colombia', 'United States East Coast', 'Atlantic Canada'],
                  ['Cape Verde', 'The Caribbean', 'British Virgin Islands', 'U.S. Virgin Islands', 'Cuba', 'Florida'],
                  ['Lesser Antilles', 'Virgin Islands', 'Puerto Rico', 'Dominican Republic',
                   'Turks and Caicos Islands'],
                  ['Central America', 'United States Gulf Coast (especially Florida Panhandle)']]

# damages (USD($)) of hurricanes
damages = ['Damages not recorded', '100M', 'Damages not recorded', '40M', '27.9M', '5M', 'Damages not recorded', '306M',
           '2M', '65.8M', '326M', '60.3M', '208M', '1.42B', '25.4M', 'Damages not recorded', '1.54B', '1.24B', '7.1B',
           '10B', '26.5B', '6.2B', '5.37B', '23.3B', '1.01B', '125B', '12B', '29.4B', '1.76B', '720M', '15.1B', '64.8B',
           '91.6B', '25.1B']

# deaths for each hurricane
deaths = [90, 4000, 16, 3103, 179, 184, 408, 682, 5, 1023, 43, 319, 688, 259, 37, 11, 2068, 269, 318, 107, 65, 19325,
          51, 124, 17, 1836, 125, 87, 45, 133, 603, 138, 3057, 74]

# 1
# Update Recorded Damages
conversion = {"M": 1000000,
              "B": 1000000000}

# test function by updating damages
a = 0

for i in damages:
    if i == 'Damages not recorded':
        damages[a] = i
        a += 1
        continue
    elif 'M' in str(i):
        damages[a] = float(i.split('M')[0]) * float(conversion['M'])
        a += 1
    elif 'B' in str(i):
        damages[a] = float(i.split('B')[0]) * float(conversion['B'])
        a += 1

# 2 
# Create a Table
hurricane = {}
ara = {}

# Create and view the hurricanes dictionary

for i in range(0, len(names)):
    ara['Name'] = names[i]
    ara['Month'] = months[i]
    ara['Year'] = years[i]
    ara['Max Sustained Wind'] = max_sustained_winds[i]
    ara['Areas Affected'] = areas_affected[i]
    ara['Damage'] = damages[i]
    ara['Death'] = deaths[i]
    hurricane[names[i]] = ara
    ara = {}

# 3
# Organizing by Year

for i in range(0, len(years)):
    hurricane[years[i]] = hurricane.pop(names[i])

# print(hurricane)

# create a new dictionary of hurricanes with year and key


# 4
# Counting Damaged Areas
count = 0
affected = {}
for i in hurricane.keys():
    for j in hurricane[i]['Areas Affected']:
        if j not in affected.keys():
            affected[j] = 1
        elif j in affected.keys():
            affected[j] = affected[j] + 1

# print(affected)

# count = { k: len(v) for k, v in hurricane.items() if k == 'Areas Affected'}

# create dictionary of areas to store the number of hurricanes involved in

# 5 
# Calculating Maximum Hurricane Count
# find most frequently affected area and the number of hurricanes involved in

print('The most hurricanes happen ' + str(max(affected.items(), key=operator.itemgetter(1))[1]) + ' times in ' + str(
    max(affected.items(), key=operator.itemgetter(1))[0]))

# 6
# Calculating the Deadliest Hurricane

maks_value = 0

for i in hurricane.keys():
    if hurricane[i]['Death'] > maks_value:
        maks_value = hurricane[i]['Death']
        hname = hurricane[i]['Name']
    else:
        continue

print('Maximum Death is ' + str(maks_value) + ' in ' + str(hname))


# find highest mortality hurricane and the number of deaths

# 7
# Rating Hurricanes by Mortality

# categorize hurricanes in new dictionary with damage severity as key
def max_areas_affected(areas_count):
    max_area = ''
    max_count = 0


def fatality(hurricane):
    hurricane_most_deaths = ''
    number_of_deaths = 0

    for i in hurricane:
        if hurricane[i]['Death'] > number_of_deaths:
            hurricane_most_deaths = i
            number_of_deaths = hurricane[i]['Death']
    return number_of_deaths, hurricane_most_deaths


most_deaths, number_of_deaths = fatality(hurricane)
print(most_deaths, number_of_deaths)


# Categorize hurricanes by mortality rates and return a dictionary.

def mortality(hurricane):
    mortality_rates = {0: [], 1: [], 2: [], 3: [], 4: []}

    for jj in hurricane:

        rate = 0
        deaths = hurricane[jj]['Death']

        if deaths < 100:
            rate = 0
        elif deaths >= 100 and deaths < 500:
            rate = 1
        elif deaths >= 500 and deaths < 1000:
            rate = 2
        elif deaths >= 1000 and deaths < 10000:
            rate = 3
        else:
            rate = 4

        if rate not in mortality_rates:
            mortality_rates[rate] = hurricane[jj]
        else:
            mortality_rates[rate].append(hurricane[jj])

    return mortality_rates


mortality_rates = mortality(hurricane)


# print(mortality_rates)

# Find the highest damage inducing hurricane and its total cost.

def max_damage(hurricane):
    max_damage_hurricane = ''
    max_damage_number = 0

    for k in hurricane:
        if hurricane[k]['Damage'] == 'Damages not recorded':
            continue
        if hurricane[k]['Damage'] > max_damage_number:
            max_damage_hurricane = hurricane[k]['Name']
            max_damage_number = hurricane[k]['Damage']
    return max_damage_hurricane, max_damage_number


max_damage_hurricane, max_damage_number = max_damage(hurricane)


# print(max_damage_hurricane, max_damage_number)

# Categorize hurricanes by damage rates and return a dictionary

def damage_scaled(hurricane):
    damage_scale = {0: [], 1: [], 2: [], 3: [], 4: []}

    for z in hurricane:

        rate = 0
        damage = hurricane[z]['Damage']

        if damage == 'Damages not recorded':
            continue
        elif damage < 100000000:
            rate = 0
        elif damage >= 100000000 and damage < 1000000000:
            rate = 1
        elif damage >= 1000000000 and damage < 10000000000:
            rate = 2
        elif damage >= 10000000000 and damage < 50000000000:
            rate = 3
        else:
            rate = 4

        if rate not in damage_scale:
            damage_scale[rate] = hurricane[z]
        else:
            damage_scale[rate].append(hurricane[z])

    return damage_scale


damage_scale = damage_scaled(hurricane)
# print(damage_scale)