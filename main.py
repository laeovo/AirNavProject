import numpy as np
import math
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

earthRadius = 6371000 # meters

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
    rotationLon = np.array([[math.cos(degToRad(originLlh[1])), -math.sin(degToRad(originLlh[1])), 0],
                            [math.sin(degToRad(originLlh[1])), math.cos(degToRad(originLlh[1])), 0],
                            [0, 0, 1]])
    rotationLat = np.array([[math.cos(degToRad(originLlh[0])), 0, -math.sin(degToRad(originLlh[0]))],
                            [0, 1, 0],
                            [math.sin(degToRad(originLlh[0])), 0, math.cos(degToRad(originLlh[0]))]])
    rotationCoordinates = np.array([[0, 0, 1],
                                    [1, 0, 0],
                                    [0, 1, 0]])
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

# loop through grid points
for i in range(gridLon.shape[0]):
    print("progress:", i+1, "/", gridLon.shape[0])
    for j in range(gridLon.shape[1]):
        lat = gridLat[i, j]
        lon = gridLon[i, j]
        userLlh = np.array([lat, lon, feetToMeters(10000)])
        # loop through DMEs in data list
        for row in range(len(data)):
            dmeLlh = np.array([data[row, 2], data[row, 3], 0])
            dmeRange = computeDmeRange(data[row, 4])
            if distanceLlh(userLlh, dmeLlh) <= dmeRange:
                values[i, j] += 1 / distanceLlh(userLlh, dmeLlh)
# values = np.random.uniform(0, 1, gridLon.shape)

# prepare the plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([westEnd, eastEnd, southEnd, northEnd], crs=ccrs.PlateCarree())

# draw europe
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# plot the colored graph to show expected position accuracy
plt.scatter(gridLon, gridLat, s=2, c=values)
plt.set_cmap("RdYlGn")
plt.colorbar()

# plot DMEs
plt.scatter(data[:,3], data[:,2], marker="x")

plt.show()
