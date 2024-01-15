import numpy as np
import math
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def metersToMiles(lengthInMeters):
    return lengthInMeters / 1852.0

def milesToMeters(lengthInMiles):
    return lengthInMiles * 1852.0

def feetToMeters(feet):
    return feet * 0.3048

def degToRad(deg):
    return deg / 180.0 * math.pi

def radToDeg(rad):
    return rad / math.pi * 180.0

earthRadius = 6371000 # meters
sigma_dme = milesToMeters(0.2)/2 # two sigmas (95%) would be 0.2 NM
sigma_altimeter = feetToMeters(100) / 2 # two sigmas (95%) would be 100 ft

# lat/lon/height (deg, m amsl) to ecef (m)
def llhToEcef(llh):
    r = llh[2] + earthRadius
    x = r * math.cos(degToRad(llh[0])) * math.cos(degToRad(llh[1]))
    y = r * math.cos(degToRad(llh[0])) * math.sin(degToRad(llh[1]))
    z = r * math.sin(degToRad(llh[0]))
    return np.array([x, y, z])

# ecef (m) to lat/lon/height (deg, m amsl)
def ecefToLlh(ecef):
    if ecef[0] == 0 and ecef[1] == 0:
        if ecef[2] > 0: lat = 90
        elif ecef[2] < 0: lat = -90
        else: lat = 0
    else:
        lat = radToDeg(math.atan(ecef[2] / math.sqrt(ecef[0]*ecef[0]+ecef[1]*ecef[1])))
    lon = radToDeg(math.atan2(ecef[1], ecef[0]))
    height = math.sqrt(ecef[0]*ecef[0]+ecef[1]*ecef[1]+ecef[2]*ecef[2]) - earthRadius
    return np.array([lat, lon, height])

# ecef to enu, given origin (ecef)
def ecefToEnu(ecef, originEcef):
    originLlh = ecefToLlh(originEcef)
    rotationLon = np.array([[math.cos(degToRad(-originLlh[1])), -math.sin(degToRad(-originLlh[1])), 0],
                            [math.sin(degToRad(-originLlh[1])), math.cos(degToRad(-originLlh[1])), 0],
                            [0, 0, 1]])
    rotationLat = np.array([[math.cos(degToRad(-originLlh[0])), 0, -math.sin(degToRad(-originLlh[0]))],
                            [0, 1, 0],
                            [math.sin(degToRad(-originLlh[0])), 0, math.cos(degToRad(-originLlh[0]))]])
    rotationCoordinates = np.array([[0, 1, 0],
                                    [0, 0, 1],
                                    [1, 0, 0]])
    return rotationCoordinates@rotationLat@rotationLon@(ecef-originEcef)

# enu to ecef, given origin (ecef)
def enuToEcef(enu, originEcef):
    originLlh = ecefToLlh(originEcef)
    rotationCoordinates = np.array([[0, 0, 1],
                                    [1, 0, 0],
                                    [0, 1, 0]])
    rotationLat = np.array([[math.cos(degToRad(originLlh[0])), 0, -math.sin(degToRad(originLlh[0]))],
                            [0, 1, 0],
                            [math.sin(degToRad(originLlh[0])), 0, math.cos(degToRad(originLlh[0]))]])
    rotationLon = np.array([[math.cos(degToRad(originLlh[1])), -math.sin(degToRad(originLlh[1])), 0],
                            [math.sin(degToRad(originLlh[1])), math.cos(degToRad(originLlh[1])), 0],
                            [0, 0, 1]])
    return originEcef+rotationLon@rotationLat@rotationCoordinates@enu

# distance between two coords given in ecef (m)
def distanceEcef(a, b):
    return math.sqrt((a[0] - b[0]) * (a[0] - b[0]) +
                     (a[1] - b[1]) * (a[1] - b[1]) +
                     (a[2] - b[2]) * (a[2] - b[2]))

# distance between two coords given in lat/lon/height (deg, m amsl) in m
def distanceLlh(a, b):
    aEcef = llhToEcef(a)
    bEcef = llhToEcef(b)
    return distanceEcef(aEcef, bEcef)

# compute range (m) of a DME, given its EIRP (dBW)
def computeDmeRange(eirp):
    if eirp >= 30: return milesToMeters(100)
    else: return milesToMeters(50)


# get the data
data = np.loadtxt("dme.dat")
data = np.unique(data, axis=0) # remove duplicates
allDMEsCounter = len(data)

# country of choice: Czech republic (9)
northEnd = 52 # 51.052915° N
eastEnd = 19 # 18.864830° E
southEnd = 48 # 48.551768° N
westEnd = 12 # 12.091046° E

# initialize grid
gridSize = 0.05
gridCoordinatesLat = np.arange(southEnd, northEnd, gridSize)
gridCoordinatesLon = np.arange(westEnd, eastEnd, gridSize)
gridLon, gridLat = np.meshgrid(gridCoordinatesLon, gridCoordinatesLat)
values = np.zeros(gridLon.shape)

# remove DMEs that are irrelevant for this study (i.e. outside the area of interest)
deleteCounter = 0
# first delete all DMEs whose coordinates are obviously outside the area of interest
for row in np.arange(len(data)-1, 0, -1):
    DMELat = data[row, 2]
    DMELon = data[row, 3]
    # earth's circumference at the northernmost point of Europe (ca. 70° latitude) is 13,680 km.
    # 1° of longitude corresponds to 38 km
    # the strongest DMEs have a range of 100 NM (185.2 km), which equates to maximum 4.87° of longitude.
    # let's drop all DMEs that lie 5° or greater outside the area of interest in at least one direction
    threshold = 5
    if DMELat >= northEnd + threshold or DMELon >= eastEnd + threshold or DMELat <= southEnd - threshold or DMELon <= westEnd - threshold:
        data = np.delete(data, row, 0)
        deleteCounter += 1

# now check for each DME if it is received at at least one gridpoint
print("Preprocessing: discarding all DMEs outside area of interest")
nrDMEsAfterFirstPreprocessing = len(data)
for row in np.arange(len(data)-1, 0, -1):
    progress = nrDMEsAfterFirstPreprocessing - row
    if progress % 10 == 0:
        print("Progress:", progress,"/", nrDMEsAfterFirstPreprocessing)
    dmeReceived = False
    dmeLlh = np.array([data[row, 2], data[row, 3], 0])
    dmeEcef = llhToEcef(dmeLlh)
    dmeRange = computeDmeRange(data[row, 4])
    for i in range(gridLon.shape[0]):
        for j in range(gridLon.shape[1]):
            lat = gridLat[i, j]
            lon = gridLon[i, j]
            userLlh = np.array([lat, lon, feetToMeters(10000)])
            userEcef = llhToEcef(userLlh)
            if distanceEcef(dmeEcef, userEcef) <= dmeRange:
                dmeReceived = True
                break
        if dmeReceived: break
    if not dmeReceived:
        data = np.delete(data, row, 0)
        deleteCounter += 1
print("Removed", deleteCounter, "DMEs (previously", allDMEsCounter, "DMEs) from list because they are too far from the area of interest.")

# loop through grid points
print()
print("Now computing HDOP at each gridpoint")
for i in range(gridLon.shape[0]):
    print("progress:", i+1, "/", gridLon.shape[0])
    for j in range(gridLon.shape[1]):
        lat = gridLat[i, j]
        lon = gridLon[i, j]
        userLlh = np.array([lat, lon, feetToMeters(10000)])
        userEcef = llhToEcef(userLlh)
        H = np.array([])
        # loop through DMEs in data list
        for row in range(len(data)):
            dmeLlh = np.array([data[row, 2], data[row, 3], 0])
            dmeEcef = llhToEcef(dmeLlh)
            dmeENU = ecefToEnu(dmeEcef, userEcef)
            dmeENUUnitVector = dmeENU / np.linalg.norm(dmeENU)
            dmeRange = computeDmeRange(data[row, 4])
            # check if DME signal is received by user
            if distanceLlh(userLlh, dmeLlh) <= dmeRange:
                if len(H) == 0: H = np.array([dmeENUUnitVector])
                else: H = np.concatenate((H, np.array([dmeENUUnitVector])), axis=0)

        if len(H) >= 3:
            # compute HDOP

            # covRho = np.diag((np.repeat(sigma_dme*sigma_dme, len(H)))) # not needed to compute HDOP
            G = np.linalg.inv(H.T@H).T
            HDOP = math.sqrt(G[0, 0] + G[1, 1])

            if HDOP > 10: values[i, j] = np.nan
            else: values[i, j] = HDOP

        else:
            # not enough DMEs received, report HDOP as 'not a number'
            values[i, j] = np.nan






# prepare the plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([westEnd, eastEnd, southEnd, northEnd], crs=ccrs.PlateCarree())

# draw europe
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# plot the colored graph to show expected position accuracy
plt.pcolormesh(gridLon, gridLat, values, cmap="RdYlGn_r")
plt.colorbar()

# plot DMEs
plt.scatter(data[:, 3], data[:, 2], marker="x")

plt.show()
