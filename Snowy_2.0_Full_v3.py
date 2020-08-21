import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

colnames = ['time', 'wind', 'pv', 'demand', 'price']
n = 105120

df = pd.read_csv('full_data.csv', names=colnames, nrows=n)

time_list = df.time.tolist()
wind_list = df.wind.tolist()
pv_list = df.pv.tolist()
demand_list = df.demand.tolist()
price_list = df.price.tolist()

'''
thought logic:

CHARGING
IF spot price > operating cost
    generator sell to market 
ELSE
    sell to pumped hydro for x amount (look at PPAs?)

DISCHARGING
IF spot price > y (some strike price, can be varied)
    sell to market 
ELSE 
    store energy 

CONSTRAINTS
- maximum stored capacity 
- maximum power charge + discharge?
- ~40% loss on discharge for pumped hydro

'''
# Financial Calculations

# e.g. PV

peak_load = 1
pv_ratio = 0.5
wind_ratio = 1 - pv_ratio

f_pv = 150000  # capital + O&M ($/MW/yr)
v_pv = 0  # SRMC ($/MWh)
cap_pv = 20  # installed capacity (MW)

# e.g. Wind

f_wind = 230000  # capital + O&M ($/MW/yr)
v_wind = 5  # SRMC ($/MWh)
cap_wind = 140.7 # installed capacity (MW)

# ---------------------------------
# Storage

# f_hydro = 0 #100000  # capital + O&M ($/MW/yr)
v_hydro = 2  # SRMC ($/MWh)
cap_hydro = 150  # installed capacity (MW)
max_cap = 350000000
hydro_storage = 1600
k = 0.03 # interest rate
m = 20 # lifetime
f_hydro = 4000000000 * hydro_storage/max_cap * (k * (1+k) ** m ) / ((k+1)**m -1)

f_battery = 200000  # capital + O&M ($/MW/yr)
v_battery = 0  # SRMC ($/MWh)
cap_battery = 150  # installed capacity (MW)

# ---------------------------------
# logic

''' estimating that hydro could set aside ~ 100MW capacity for both RE generators '''

strike_price = 10 # price that hydro will charge
discharge_price = 100 # price that hydro will discharge
strike_price_step = 5
discharge_price_step = 5
hydro_discharge_max_rate = 2000 # maximum discharge rate 

op_profit_pv = np.zeros(n)
op_profit_wind = np.zeros(n)
op_profit_hydro = np.zeros(n)
op_cost_hydro = np.zeros(n)
stored_hydro = np.zeros(n)
profit_pv_list = []
profit_wind_list = []
profit_hydro_list = []
strike_price_list = []
discharge_price_list = []


gen_pv = pv_list  # change to pv gen data
gen_wind = wind_list  # change to wind gen data

pv_profit_storage = np.zeros(n)
storage_soc = np.zeros(n)
soc_hydro = 0
time_diff = 5
min_in_hour = 60

hydro_discharge_max = hydro_discharge_max_rate * (time_diff/min_in_hour) # in MWh


while discharge_price <= 270:
    for i in range(n):
        if i>0 : # exclude the first row of data 
            # check to see if its charging or discharging

            # discharging limited potentially to every hour for pumped hydro?
            if price_list[i] >= strike_price : # if spot price > 10
                
                if price_list[i] >= discharge_price : # if spot price > 100
                    # hydro
                    if storage_soc[i-1] > hydro_discharge_max and price_list[i-1] > discharge_price: # if the hydro has the capacity to discharge at max and if
                                                                                                # price in the interval before this is > than discharge price
                                                                                                # to mimic hydro lag
                        # check = check + 1
                        op_profit_hydro[i] = hydro_discharge_max * (price_list[i] - v_hydro)
                        storage_soc[i] = storage_soc[i-1] - hydro_discharge_max

                    elif storage_soc[i-1] <= hydro_discharge_max and price_list[i-1] > discharge_price : # if hydro can't discharge at full capacity
                        op_profit_hydro[i] = storage_soc[i-1] * (price_list[i] - v_hydro)
                        storage_soc[i] = 0

                    elif price_list[i-1] < discharge_price : # if spot price in the interval before is less than discharge price -> stop discharging
                        storage_soc[i] = storage_soc[i-1]

                    op_profit_pv[i] = gen_pv[i] * (price_list[i] - v_pv) # $/hr
                    op_profit_wind[i] = gen_wind[i] * (price_list[i] - v_wind)

                    # # battery
                    # if storage_soc[i-1] > hydro_discharge_max :
                    #     op_profit_hydro[i] = hydro_discharge_max * price_list[i] - v_hydro
                    #     storage_soc[i] = storage_soc[i-1] - hydro_discharge_max
                    #
                    # elif storage_soc[i-1] <= hydro_discharge_max :
                    #     op_profit_hydro[i] = storage_soc[i-1] * (price_list[i] - v_hydro)
                    #     storage_soc[i] = 0
                    #
                    # op_profit_pv[i] = gen_pv[i] * (price_list[i] - v_pv) # $/hr
                    # op_profit_wind[i] = gen_wind[i] * (price_list[i] - v_wind)

                else : # if spot price < 100 and > 10
                    
                    # generator sell to market
                    op_profit_pv[i] = gen_pv[i] * (price_list[i] - v_pv) # $/hr
                    op_profit_wind[i] = gen_wind[i] * (price_list[i] - v_wind)
                    op_profit_hydro[i] = 0
                    stored_hydro[i] = 0
                    storage_soc[i] = storage_soc[i-1]

            else : # if price_list[i] < strike_price 
                # generator sell to hydro
                checker = 0 # to reset the checker, and make sure that the hydro stops discharging.
                op_profit_pv[i] = gen_pv[i] * (strike_price - v_pv)
                op_profit_wind[i] = gen_wind[i] * (strike_price - v_wind)
                op_profit_hydro[i] = - (strike_price * (gen_pv[i] + gen_wind[i]))
                stored_hydro[i] = gen_pv[i] + gen_wind[i]
                
                storage_soc[i] = storage_soc[i-1] + (stored_hydro[i] * (time_diff/min_in_hour)) #keep SOC as MWh for now, we'll normalise it later

        else : 
            storage_soc[i] = 0

    # revenue_pv = 24 * 365 * sum(op_profit_pv) / n
    # revenue_wind = 24 * 365 * sum(op_profit_wind) / n
    # revenue_hydro = 24 * 365 * sum(op_profit_hydro) / n

    profit_pv = (24 * 365 * sum(op_profit_pv) / n - f_pv * cap_pv)/1000 # $k/yr
    profit_pv_list.append(profit_pv)

    profit_wind = (24 * 365 * sum(op_profit_wind) / n - f_wind * cap_wind)/1000 # $k/yr
    profit_wind_list.append(profit_wind)

    profit_hydro = 0.6 * (24 * 365 * sum(op_profit_hydro * 12) / n - f_hydro * cap_hydro)/1000 # X 12 cos op profit hydro was in $/5min unit
    profit_hydro_list.append(profit_hydro)

    strike_price_list.append(strike_price)
    discharge_price_list.append(discharge_price)

    # strike_price = strike_price + strike_price_step
    discharge_price = discharge_price + discharge_price_step

# print(sum(op_profit_hydro))
# print('Wind Profit:', profit_wind_list)
# print('PV Profit:', profit_pv_list)
# print('Hydro Profit:', profit_hydro_list)

# plt.plot(strike_price_list, profit_pv_list)
# plt.plot(strike_price_list, profit_wind_list)
# plt.plot(strike_price_list, profit_hydro_list)
# plt.ylabel('Profit ($m/yr)')
# plt.xlabel('Strike Price ($/MWh)')
# plt.legend(["PV Profit", "Wind Profit", "Hydro Profit"])
# plt.show()

# plt.plot(discharge_price_list, profit_pv_list)
# plt.plot(discharge_price_list, profit_wind_list)
plt.plot(discharge_price_list, profit_hydro_list)
plt.ylabel('Profit ($k/yr)')
plt.xlabel('Discharge Price ($/MWh)')
plt.legend(["Hydro Profit"])
plt.show()

days = np.linspace(0, 365, n)
plt.plot(days, storage_soc)
plt.xlabel('Day')
plt.ylabel('Battery charge (MWh)')
plt.show()

# print(sum(op_profit_pv))
# print(sum(op_profit_wind))
# print(sum(op_profit_hydro))
# print(sum(op_cost_hydro))

# revenue_pv = 24 * 365 * sum(op_profit_pv) / n
# print('PV Revenue:', revenue_pv)

# revenue_wind = 24 * 365 * sum(op_profit_wind) / n
# print('Wind Revenue:', revenue_wind)

# profit_pv = 24 * 365 * sum(op_profit_pv) / n - f_pv * cap_pv # $/yr
# print('PV Profit:', profit_pv)

# profit_wind = 24 * 365 * sum(op_profit_wind) / n - f_wind * cap_wind # $/yr
# print('Wind Profit:', profit_wind)

# revenue_hydro = 24 * 365 * sum(op_profit_hydro) / n
# print('Hydro Revenue:', revenue_hydro)

# profit_hydro = 24 * 365 * sum(op_profit_hydro) / n - f_hydro * cap_hydro # $/yr
# print('Hydro Profit:', profit_hydro)

# plt.plot(op_profit_pv)
# plt.show()


# plt.plot(gen_pv)
# plt.plot(gen_wind)
# plt.plot(stored_hydro)
# plt.plot(op_profit_wind)
# plt.show()

