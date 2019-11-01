# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:57:29 2019

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:17:23 2019

@author: Administrator
"""

#Global imports

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import requests
import json
import numpy as np

def get_data_request(url, requestData, access_token):
    '''make HTTP GET request'''
    dResp = requests.get(url, headers = {'X-api-key': access_token}, params = requestData);       

    
    if dResp.status_code != 200:
        print("Unable to get data. Code %s, Message: %s" % (dResp.status_code, dResp.text));
    else:
        print("Data access successful")
        jResp = json.loads(dResp.text);
        return jResp

def get_historical_price(rics, start_date, end_date, fields="", debug_printing=False):
    
    output_df = None
    for ric in rics:
    
        RESOURCE_ENDPOINT = "https://dsa-stg-edp-api.fr-nonprod.aws.thomsonreuters.com/data/historical-pricing/beta1/views/summaries/" + ric
        access_token = 'KXH0s802dI8jOZ1MKDQz72ligJMKwfo58GX2LtFf'  # your personal key for Data Science Accelerator access to Pricing Data
      
        requestData = {
            "interval": "P1D",
            "start": start_date,
            "end": end_date,
            #"fields": 'TRDPRC_1' #BID,ASK,OPEN_PRC,HIGH_1,LOW_1,TRDPRC_1,NUM_MOVES,TRNOVR_UNS
        }
        
        if fields:
            requestData["fields"] = fields
    
        jResp = get_data_request(RESOURCE_ENDPOINT, requestData, access_token)
        
        if debug_printing:
            print(jResp)
        
        if jResp is not None and 'data' in jResp[0]:
            data = jResp[0]['data']
            headers = jResp[0]['headers']  
            names = [headers[x]['name'] for x in range(len(headers))]
            
            spot_df = pd.DataFrame(data, columns=names)
            spot_df = spot_df.set_index( pd.to_datetime(spot_df['DATE']))
            spot_df = spot_df.drop(columns=["DATE"])
            
            # Combine BID / ASK
            if fields == "BID, ASK":
                spot_df["CLOSE"] = ( spot_df["BID"] + spot_df["ASK"] ) / 2
                spot_df = spot_df.drop(columns=["BID","ASK"])
                spot_df.columns = [ric]
                
#                returns = spot_df.pct_change()
#                
#                returns = returns.where((returns < returns.mean() - returns.std() * 6)) | (returns > returns.mean() + returns.std() * 6)
#                
#                spot_df = spot_df.where
                
#                spot_df.columns = [RIC_to_CCY(ric)]
                
            if fields == "MID_YLD_1":
                spot_df.columns = [ric]
                pass
            
               
            if output_df is None:
                output_df = spot_df
            else:
                output_df = output_df.join(spot_df, how="outer")
    
    return output_df

def CCY_to_RIC(ccy):
    return "={0}_TRDPRC_1".format(ccy)

def RIC_to_CCY(ric):
    return ric[:3]

def get_relevant_CPI(CPI_filename="CPI.csv", frequency = "M", countries_to_collect=[]):

    cpi = pd.read_csv(CPI_filename)
    
    # Extract monthly CPI for relevant currencies
    monthly_cpi = cpi[ (cpi["FREQUENCY"] == frequency) & (cpi["LOCATION"].isin(countries_to_collect)) ]
    
    return monthly_cpi


########################### Define Constants ################################
country_to_CCY_dict = {
        "AUS":"AUD",
        "CAN":"CAD",
        "CHE":"CHF",
        "DEU":"EUR",        # To replace with a basket of currencies
        "GBR":"GBP",
        "JPN":"JPY",
        "NOR":"NOK",
        "NZL":"NZD",
        "SWE":"SEK",
        "USA":"USD",
        }

CCY_to_country_dict = {
        "AUD":"AUS",
        "CAD":"CAN",
        "CHF":"CHE",
        "EUR":"DEU",
        "GBP":"GBR",
        "JPY":"JPN",
        "NOK":"NOR",
        "NZD":"NZL",
        "SEK":"SWE",
        "USD":"USA",
        }

G10_ccys = ["AUD", "CAD","CHF","EUR","GBP","JPY","NOK","NZD","SEK","USD"]
G10_country_codes = ["AUS","CAN","CHE","DEU","GBR","JPN","NOR","NZL","SWE","USA"]

FX_spot_filename = "FX_spot.csv"
CPI_filename     = "CPI.csv"
FUT_filename     = "FX_FUT.csv"

data_folder      = "first_round/data"

if __name__ == "__main__":
#    spot_df = get_historical_price([ CCY + "=" for CCY in G10_ccys], "2016-11-01", "2019-06-30", fields="BID, ASK")
#    
#    spot_df["GBP"] = 1/ spot_df["GBP"]
#    spot_df["EUR"] = 1/spot_df["EUR"]
#    
#    spot_df.to_csv(FX_spot_filename)
    
#    fut_df = get_historical_price([ CCY + "3MV=" for CCY in G10_ccys], "2016-11-01", "2019-06-30", fields="BID, ASK", debug_printing=False)
#    
#    fut_df["AUD"] = 1/fut_df["AUD"]
#    fut_df["EUR"] = 1/fut_df["EUR"]
#    fut_df["GBP"] = 1/fut_df["GBP"]
#    fut_df["NZD"] = 1/fut_df["NZD"]
#    
#    fut_df.to_csv(FUT_filename)

    test = get_historical_price(["SDY"],"2007-01-01","2019-07-31", fields="BID, ASK", debug_printing=False)
    
    