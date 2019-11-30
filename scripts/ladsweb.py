from SOAPpy import SOAPProxy
import time, os, argparse
import json
import pdb 

__all__ = ['search_parameters', 'post_processing', 'bands', 'getData']

search_parameters = {
    'products': 'NPP_VMAES_L1', 
    'startTime': '2018-07-04',
    'endTime': '2018-07-05',
    'north': 44,
    'south': 43, 
    'east': -6, 
    'west': -7, 
    'coordsOrTiles': 'coords',
    'dayNightBoth': 'D'
}

post_processing = {
    'reprojectionName': 'GEO',
    'reprojectionOutputPixelSize': '0.01',
    'reprojectionResampleType': 'bilinear',
    'doMosaic': False
}

bands =  [
    'Reflectance_M5', 
    'Reflectance_M7', 
    'Reflectance_M10', 
    'Radiance_M12', 
    'Radiance_M15',
    'SolarZenithAngle', 
    'SatelliteZenithAngle'
]

def getData(email, auth, path, search_parameters, post_processing, bands, release_order=True,
            request_max_wait_time=60*24, download_max_wait_time=60):

    url = 'https://modwebsrv.modaps.eosdis.nasa.gov/axis2/services/MODAPSservices?wsdl'

    client = SOAPProxy(url)

    print('Searching for files')
    response = client.searchForFiles(**search_parameters)

    # Formating band names  
    bandStr = ','.join([search_parameters['products'] + '___' + b for b in bands]) 

    # Requesting files
    orderID = client.orderFiles(
        email=email, fileIds=','.join(response),
        geoSubsetNorth=search_parameters['north'],
        geoSubsetSouth=search_parameters['south'],
        geoSubsetWest=search_parameters['west'],
        geoSubsetEast=search_parameters['east'],        
        subsetDataLayer=bandStr, **post_processing)

    status = client.getOrderStatus(orderId=orderID)
    print('The order is submitted with the ID: ' + orderID)

    # Waiting for the request to be processed
    i = 0
    while status != 'Available' and i < request_max_wait_time: 
        print('Request status: ' + status)
        time.sleep(60)
        status = client.getOrderStatus(orderId=orderID)
        i += 1
    if i >= request_max_wait_time:
        raise Exception('request_max_wait_time exceeded')
    else:
        order_url = client.getOrderUrl(orderId=orderID)
        print('Request concluded. Starting data download from ' + order_url)

    command = ('wget -e robots=off -m -np -R .html,.tmp -nH --cut-dirs=3'
            ' https://ladsweb.modaps.eosdis.nasa.gov/archive/orders/' 
            + orderID + '/ --header "Authorization: Bearer ' 
            + auth + '" -P ' + path)

    # Downloading files
    download, i = 1, 0
    while download != 0 and i < download_max_wait_time: 
        print('Waiting 5 minutes to start/restart download')
        time.sleep(5*60)
        print('Starting download...')
        download = os.system(command)
        i += 1
    if i >= download_max_wait_time:
        raise Exception('download_max_wait_time exceeded')
    else:
        print('Data download finished sucessfully!')
        if release_order:
            client.releaseOrder(orderId=orderID, email=email)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('email', type=str)
    arg('auth', type=str)
    arg('path', type=str)
    arg('region', type=str)
    arg('tstart', type=str)
    arg('tend', type=str)
    args = parser.parse_args()

    search_parameters['startTime'] = args.tstart
    search_parameters['endTime'] = args.tend

    with open('../data/regions/R_' + args.region + '.json', 'r') as f:
        R = json.load(f)

    search_parameters['north'] = R['bbox'][3]
    search_parameters['south'] = R['bbox'][1]
    search_parameters['west'] = R['bbox'][0]
    search_parameters['east'] = R['bbox'][2]

    post_processing['reprojectionOutputPixelSize'] = str(R['pixel_size'])

    # Call getData with args
    getData(args.email, args.auth, args.path, search_parameters, post_processing, bands)