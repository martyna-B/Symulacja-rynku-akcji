import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
import csv
import os
import math
import numpy as np
import random

def take_data(my_file, index):
    """
    Returns a list which contains particular data from my_file.

    Paramters
    ---------
    my_file(file)
    """
    if os.path.isfile(my_file):
        with open(my_file, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            lines = []
            for line in csv_reader:
                try:
                    lines.append(float(line[index]))
                except:
                    lines.append(line[index])
            del lines[0]
        return lines
    else:
        raise FileNotFoundError

def data_plot(file_name, plot_title, time, inter):
    """
    Creates a plot with datas on the x axis.
    """
    if not os.path.isfile(file_name):
        raise FileNotFoundError
    if type(plot_title) != str or type(time) != int or type(inter) != int:
        raise TypeError
        
    file_data = take_data(file_name, 1)
    file_times = take_data(file_name, 0)
    times = [dt.datetime.strptime(data[:10],'%Y-%m-%d').date() for data in file_times[time:]]
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = inter))
    plt.plot(times, file_data[time:])
    plt.gcf().autofmt_xdate()
    plt.title(plot_title)
    plt.show()
    
def oblicz_równanie(a, b, c, d, x_0, y_0):
    """
    Wylicza współczynniki do równania obliczonego analitycznie.
    """
    g = a - b
    f = d - c
    r_1 = (g + f + math.sqrt((g-f)**2 + 4*b*c))/2
    r_2 = (g + f - math.sqrt((g-f)**2 + 4*b*c))/2
    C_1 = (c*y_0 - x_0*(r_2 - g))/(r_1 - r_2)
    C_2 = (x_0*(r_1 - g) - c*y_0)/(r_1 - r_2)
    y_const_1 = (1/c)*((r_1 - g)*C_1)
    y_const_2 = (1/c)*((r_2 - g)*C_2)
    return "Y const 1: %s, Y const 2: %s, X conts 1: %s, X const 2: %s, r_1: %s, r_2: %s" % (y_const_1, y_const_2, C_1, C_2, r_1, r_2)

def wykres_numerycznie1(sinusy = False, sin_rand = False, sp = False, sin_rand_sp_events = False):
    """
    Generuje wykres dla rozwiązania numerycznego dla parametrów: sin(t/50)
    i -sin(t/50).

    Argumenty
    ---------
    sinusy(bool): jeśli True, generuje się wykres bez czynników losowych
    sin_rand(bool): jeśli True, generuje się wykres z uwzglednieniem szumu
    sp(bool): jeśli True, generuje się wykres z szumem i uwzględnieniem zmian cen akcji S&P50
    sin_rand_sp_events(bool): jeśli True, generuje się wykres z wszystkim powyższym
                                oraz znaczącymi elementami losowymi
    """

    random.seed(12345)

    sp500 = take_data("SP500.csv", 1)
    ford = take_data("Ford.csv", 1)
    GM = take_data("GM.csv", 1)

    delt_sp500 = np.diff((sp500))

    sp500 = sp500[:2358]

    if sinusy:
        ford = take_data("Ford.csv", 1)
        GM = take_data("GM.csv", 1)

        GM = GM[336:]

        ford_1 = [32]
        GM_1 = [47]

        for i in range(1,len(GM)):

            gm_r =0.0015 * math.sin(i/50)/2 
            f_r = 0.0015 * -math.sin((i/50))/2

            a = 1 + 0.00001 * random.randint(5,35) #1.00022
            b = 1 + 0.00001 * random.randint(20,45)  # 1.00035
        

            if i > 3:
                ford_1.append((ford_1[i-1]*a) - (ford_1[i-1]*gm_r - GM_1[i-1]*f_r) ) 
                GM_1.append((GM_1[i-1]*b) + (ford_1[i-1]*gm_r-GM_1[i-1]*(f_r)))
            else:
                ford_1.append(ford_1[i-1]*1.00022 )
                GM_1.append(GM_1[i-1]*1.00035 )

        plt.plot(ford_1)
        plt.plot(GM_1)
        plt.legend(['Firma X', 'Firma Y'])
        plt.title('Wykres bez elementów losowych')
        plt.show()


    if sin_rand:
        ford = take_data("Ford.csv", 1)
        GM = take_data("GM.csv", 1)

        GM = GM[336:]

        ford_1 = [32]
        GM_1 = [47]

        f_idx = 0
        g_idx = 0

        for i in range(1,len(GM)):

            gm_r =0.0015 * math.sin(i/50)/2 
            f_r = 0.0015 * -math.sin((i/50))/2

            a = 1 + 0.00001 * random.randint(5,35)  #1.00022
            b = 1 + 0.00001 * random.randint(20,45)  # 1.00035

            if i > 3:
                ford_1.append((ford_1[i-1]*a) - (ford_1[i-1]*gm_r - GM_1[i-1]*f_r)  + random.uniform(-0.25,0.25)) 
                GM_1.append((GM_1[i-1]*b) + (ford_1[i-1]*gm_r-GM_1[i-1]*(f_r)) +random.uniform(-0.25,0.25) )
            else:
                ford_1.append(ford_1[i-1]*1.00022 )
                GM_1.append(GM_1[i-1]*1.00035 )


        plt.plot(ford_1)
        plt.plot(GM_1)
        
        plt.legend(['Firma X', 'Firma Y'])
        plt.title('Wykres z nieznacznymi elementami losowymi')
        plt.show()


    if sp:
        ford = take_data("Ford.csv", 1)
        GM = take_data("GM.csv", 1)

        GM = GM[336:]

        ford_1 = [32]
        GM_1 = [47]

        f_idx = 0
        g_idx = 0

        for i in range(1,len(GM)):

            gm_r = 0.0015 * math.sin(i/50)/2
            f_r = 0.0015 * -math.sin((i/50))/2


            ford_event = f_idx*(1 + 0.00001 * random.randint(5,35))
            gm_event = g_idx*(1 + 0.00001 * random.randint(20,45))

            sp_ford = 0.01 * random.randint(3,5)
            sp_gm = 0.01 * random.randint(3,5)


            a = 1 + 0.00001 * random.randint(25,35)  #1.00022
            b = 1 + 0.00001 * random.randint(20,45)  # 1.00035

            if i > 3:
                ford_1.append((ford_1[i-1]*a) - (ford_1[i-1]*gm_r - GM_1[i-1]*f_r) + delt_sp500[i]*sp_ford  + random.uniform(-0.25,0.25) ) 
                GM_1.append((GM_1[i-1]*b) + (ford_1[i-1]*gm_r-GM_1[i-1]*(f_r))+ delt_sp500[i]*sp_gm +random.uniform(-0.25,0.25) )
            else:
                ford_1.append(ford_1[i-1]*1.00022 )
                GM_1.append(GM_1[i-1]*1.00035 )

        plt.plot(ford_1)
        plt.plot(GM_1)
        plt.axhline(y=10,xmin=0.48,xmax=0.53,color ='blue')
        plt.legend(['Firma X', 'Firma Y', 'Krach S&P500'])
        
        plt.title('Wykres z uwzględnieniem S&P500')

        plt.show()

    if sin_rand_sp_events:

        ford = take_data("Ford.csv", 1)
        GM = take_data("GM.csv", 1)

        GM = GM[336:]

        ford_1 = [320]
        GM_1 = [470]

        f_idx = 0
        g_idx = 0

        good_ford_events = []
        bad_ford_events = []

        good_gm_events = []
        bad_gm_events = []

        for i in range(1,len(GM)):

            x = random.randint(0,1000)
            y = random.randint(0,1000)
            gm_r = 0.0025 * math.sin(i/50)/4 
            f_r = 0.0025 * -math.sin((i/50))/4

            if x ==5:
                f_idx = 6
                good_ford_events.append(i) 
            elif x==3:
                f_idx = -6 #989 idx 
                bad_ford_events.append(i)
            else:
                if f_idx!=0:
                    if f_idx >0:
                        f_idx +=-0.1
                    else:
                        f_idx += 0.1
                else:
                    f_idx = 0

            if y == 4: 
                g_idx = 3  
                good_gm_events.append(i)
            elif y ==15:
                g_idx = -3
                bad_gm_events.append(i)
            else:
                if g_idx!=0:
                    if g_idx >0:
                        g_idx +=-0.1
                    else:
                        g_idx += 0.1
                else:
                    g_idx = 0    


            ford_event = f_idx*(1 + 0.00001 * random.randint(5,35))
            gm_event = g_idx*(1 + 0.00001 * random.randint(20,45))

            sp_ford = 0.08 * random.randint(3,5)
            sp_gm = 0.08 * random.randint(3,5) 

            a = 1 + 0.00001 * random.randint(15,35)  #1.00022
            b = 1 + 0.00001 * random.randint(20,45)  # 1.00035


            if i > 3:
                ford_1.append((ford_1[i-1]*a) - (ford_1[i-1]*gm_r - GM_1[i-1]*f_r) + delt_sp500[i]*sp_ford  + random.uniform(-0.25,0.25)+ford_event ) 
                GM_1.append((GM_1[i-1]*b) + (ford_1[i-1]*gm_r-GM_1[i-1]*(f_r))+ delt_sp500[i]*sp_gm +random.uniform(-0.25,0.25) + gm_event )
            else:
                ford_1.append(ford_1[i-1]*1.00022 )
                GM_1.append(GM_1[i-1]*1.00035 )


        plt.plot(ford_1)
        plt.plot(GM_1)
        for event in good_ford_events:
            plt.axvline(x=event, ymax=0.1, color = 'g')  
        for event in bad_ford_events:
            plt.axvline(x=event, ymax=0.1, color = 'r')

        plt.axhline(y=100,xmin=0.48,xmax=0.53,color ='blue')
        plt.legend(['Firma X', 'Firma Y', 'Dobre wydarzenie','Złe wydarzenie','Krach S&P500'])
        plt.title('Wykres z uwzględnieniem S&P500 i ważnych zdarzeń losowych')
        plt.show()

def wykres_numerycznie2(sinusy = False, sin_rand = False, sp = False, sin_rand_sp_events = False):
    """
    Generuje wykres dla rozwiązania numerycznego dla parametrów: sin(t/40)
    i -sin(t/70).

    Argumenty
    ---------
    sinusy(bool): jeśli True, generuje się wykres bez czynników losowych
    sin_rand(bool): jeśli True, generuje się wykres z uwzglednieniem szumu
    sp(bool): jeśli True, generuje się wykres z szumem i uwzględnieniem zmian cen akcji S&P50
    sin_rand_sp_events(bool): jeśli True, generuje się wykres z wszystkim powyższym
                                oraz znaczącymi elementami losowymi
    """
    

    random.seed(12345)

    sp500 = take_data("SP500.csv", 1)
    ford = take_data("Ford.csv", 1)
    GM = take_data("GM.csv", 1)

    delt_sp500 = np.diff((sp500))

    sp500 = sp500[:2358]

    if sinusy:
        ford = take_data("Ford.csv", 1)
        GM = take_data("GM.csv", 1)

        GM = GM[336:]

        ford_1 = [32]
        GM_1 = [47]

        for i in range(1,len(GM)):

            gm_r =0.0015 * math.sin(i/40)/2 
            f_r = 0.0015 * -math.sin((i/70))/2

            a = 1 + 0.00001 * random.randint(5,35) #1.00022
            b = 1 + 0.00001 * random.randint(20,45)  # 1.00035
        

            if i > 3:
                ford_1.append((ford_1[i-1]*a) - (ford_1[i-1]*gm_r - GM_1[i-1]*f_r) ) 
                GM_1.append((GM_1[i-1]*b) + (ford_1[i-1]*gm_r-GM_1[i-1]*(f_r)))
            else:
                ford_1.append(ford_1[i-1]*1.00022 )
                GM_1.append(GM_1[i-1]*1.00035 )

        plt.plot(ford_1)
        plt.plot(GM_1)
        plt.legend(['Firma X', 'Firma Y'])
        plt.title('Wykres bez elementów losowych')
        plt.show()


    if sin_rand:
        ford = take_data("Ford.csv", 1)
        GM = take_data("GM.csv", 1)

        GM = GM[336:]

        ford_1 = [32]
        GM_1 = [47]

        f_idx = 0
        g_idx = 0

        for i in range(1,len(GM)):

            gm_r =0.0015 * math.sin(i/40)/2 
            f_r = 0.0015 * -math.sin((i/70))/2

            a = 1 + 0.00001 * random.randint(5,35)  #1.00022
            b = 1 + 0.00001 * random.randint(20,45)  # 1.00035

            if i > 3:
                ford_1.append((ford_1[i-1]*a) - (ford_1[i-1]*gm_r - GM_1[i-1]*f_r)  + random.uniform(-0.25,0.25)) 
                GM_1.append((GM_1[i-1]*b) + (ford_1[i-1]*gm_r-GM_1[i-1]*(f_r)) +random.uniform(-0.25,0.25) )
            else:
                ford_1.append(ford_1[i-1]*1.00022 )
                GM_1.append(GM_1[i-1]*1.00035 )


        plt.plot(ford_1)
        plt.plot(GM_1)
        
        plt.legend(['Firma X', 'Firma Y'])
        plt.title('Wykres z nieznacznymi elementami losowymi')
        plt.show()


    if sp:
        ford = take_data("Ford.csv", 1)
        GM = take_data("GM.csv", 1)

        GM = GM[336:]

        ford_1 = [32]
        GM_1 = [47]

        f_idx = 0
        g_idx = 0

        for i in range(1,len(GM)):

            gm_r = 0.0015 * math.sin(i/40)/2
            f_r = 0.0015 * -math.sin((i/70))/2


            ford_event = f_idx*(1 + 0.00001 * random.randint(5,35))
            gm_event = g_idx*(1 + 0.00001 * random.randint(20,45))

            sp_ford = 0.01 * random.randint(3,5)
            sp_gm = 0.01 * random.randint(3,5)


            a = 1 + 0.00001 * random.randint(25,35)  #1.00022
            b = 1 + 0.00001 * random.randint(20,45)  # 1.00035

            if i > 3:
                ford_1.append((ford_1[i-1]*a) - (ford_1[i-1]*gm_r - GM_1[i-1]*f_r) + delt_sp500[i]*sp_ford  + random.uniform(-0.25,0.25) ) 
                GM_1.append((GM_1[i-1]*b) + (ford_1[i-1]*gm_r-GM_1[i-1]*(f_r))+ delt_sp500[i]*sp_gm +random.uniform(-0.25,0.25) )
            else:
                ford_1.append(ford_1[i-1]*1.00022 )
                GM_1.append(GM_1[i-1]*1.00035 )

        plt.plot(ford_1)
        plt.plot(GM_1)
        plt.axhline(y=10,xmin=0.48,xmax=0.53,color ='blue')
        plt.legend(['Firma X', 'Firma Y', 'Krach S&P500'])
        plt.title('Wykres z uwzględnieniem S&P500')

        plt.show()

    if sin_rand_sp_events:

        ford = take_data("Ford.csv", 1)
        GM = take_data("GM.csv", 1)

        GM = GM[336:]

        ford_1 = [320]
        GM_1 = [470]

        f_idx = 0
        g_idx = 0

        good_ford_events = []
        bad_ford_events = []

        good_gm_events = []
        bad_gm_events = []

        for i in range(1,len(GM)):

            x = random.randint(0,1000)
            y = random.randint(0,1000)
            gm_r = 0.0025 * math.sin(i/40)/4 
            f_r = 0.0025 * -math.sin((i/70))/4

            if x ==5:
                f_idx = 6
                good_ford_events.append(i) 
            elif x==3:
                f_idx = -6 #989 idx 
                bad_ford_events.append(i)
            else:
                if f_idx!=0:
                    if f_idx >0:
                        f_idx +=-0.1
                    else:
                        f_idx += 0.1
                else:
                    f_idx = 0

            if y == 4: 
                g_idx = 3  
                good_gm_events.append(i)
            elif y ==15:
                g_idx = -3
                bad_gm_events.append(i)
            else:
                if g_idx!=0:
                    if g_idx >0:
                        g_idx +=-0.1
                    else:
                        g_idx += 0.1
                else:
                    g_idx = 0    


            ford_event = f_idx*(1 + 0.00001 * random.randint(5,35))
            gm_event = g_idx*(1 + 0.00001 * random.randint(20,45))

            sp_ford = 0.08 * random.randint(3,5)
            sp_gm = 0.08 * random.randint(3,5) 

            a = 1 + 0.00001 * random.randint(15,35)  #1.00022
            b = 1 + 0.00001 * random.randint(20,45)  # 1.00035


            if i > 3:
                ford_1.append((ford_1[i-1]*a) - (ford_1[i-1]*gm_r - GM_1[i-1]*f_r) + delt_sp500[i]*sp_ford  + random.uniform(-0.25,0.25)+ford_event ) 
                GM_1.append((GM_1[i-1]*b) + (ford_1[i-1]*gm_r-GM_1[i-1]*(f_r))+ delt_sp500[i]*sp_gm +random.uniform(-0.25,0.25) + gm_event )
            else:
                ford_1.append(ford_1[i-1]*1.00022 )
                GM_1.append(GM_1[i-1]*1.00035 )


        plt.plot(ford_1)
        plt.plot(GM_1)
        for event in good_ford_events:
            plt.axvline(x=event, ymax=0.1, color = 'g')  
        for event in bad_ford_events:
            plt.axvline(x=event, ymax=0.1, color = 'r')

        plt.axhline(y=100,xmin=0.48,xmax=0.53,color ='blue')
        plt.legend(['Firma X', 'Firma Y', 'Dobre wydarzenie','Złe wydarzenie','Krach S&P500'])
        plt.title('Wykres z uwzględnieniem S&P500 i ważnych zdarzeń losowych')
        plt.show()

def wygeneruj_wykres(a, b, c, d, x_0, y_0):
    """
    Generuje wykres dla rozwiązania analitycznego układu równań różniczkowych.
    """
    g = a - b
    f = d - c
    r_1 = (g + f + math.sqrt((g-f)**2 + 4*b*c))/2
    r_2 = (g + f - math.sqrt((g-f)**2 + 4*b*c))/2
    C_1 = (c*y_0 - x_0*(r_2 - g))/(r_1 - r_2)
    C_2 = (x_0*(r_1 - g) - c*y_0)/(r_1 - r_2)
    y_const_1 = (1/c)*((r_1 - g)*C_1)
    y_const_2 = (1/c)*((r_2 - g)*C_2)
    x = [C_1*math.exp(t*r_1) + C_2 *math.exp(t*(r_2)) for t in range(0, 1000)]
    y = [y_const_1*math.exp(t*r_1) + y_const_2*math.exp(t*(r_2)) for t in range(0, 1000) ]
    plt.plot(x, label = "Ceny akcji firmy x")
    plt.plot(y, label = "Ceny akcji firmy y")
    plt.legend()
    plt.show()  
