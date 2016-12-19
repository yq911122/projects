import csv, re, json, urllib2, itertools, math
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt 
from descartes import PolygonPatch
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.cbook as cbook


def simplify_points (pts, tolerance): #from http://postgis.refractions.net/documentation/manual-svn/ST_Simplify.html
    anchor  = 0
    floater = len(pts) - 1
    stack   = []
    keep    = set()

    stack.append((anchor, floater))  
    while stack:
        anchor, floater = stack.pop()
      
        # initialize line segment
        if pts[floater] != pts[anchor]:
            anchorX = float(pts[floater][0] - pts[anchor][0])
            anchorY = float(pts[floater][1] - pts[anchor][1])
            seg_len = math.sqrt(anchorX ** 2 + anchorY ** 2)
            # get the unit vector
            anchorX /= seg_len
            anchorY /= seg_len
        else:
            anchorX = anchorY = seg_len = 0.0
    
        # inner loop:
        max_dist = 0.0
        farthest = anchor + 1
        for i in range(anchor + 1, floater):
            dist_to_seg = 0.0
            # compare to anchor
            vecX = float(pts[i][0] - pts[anchor][0])
            vecY = float(pts[i][1] - pts[anchor][1])
            seg_len = math.sqrt( vecX ** 2 + vecY ** 2 )
            # dot product:
            proj = vecX * anchorX + vecY * anchorY
            if proj < 0.0:
                dist_to_seg = seg_len
            else: 
                # compare to floater
                vecX = float(pts[i][0] - pts[floater][0])
                vecY = float(pts[i][1] - pts[floater][1])
                seg_len = math.sqrt( vecX ** 2 + vecY ** 2 )
                # dot product:
                proj = vecX * (-anchorX) + vecY * (-anchorY)
                if proj < 0.0:
                    dist_to_seg = seg_len
                else:  # calculate perpendicular distance to line (pythagorean theorem):
                    dist_to_seg = math.sqrt(abs(seg_len ** 2 - proj ** 2))
                if max_dist < dist_to_seg:
                    max_dist = dist_to_seg
                    farthest = i

        if max_dist <= tolerance: # use line segment
            keep.add(anchor)
            keep.add(floater)
        else:
            stack.append((anchor, farthest))
            stack.append((farthest, floater))

    keep = list(keep)
    keep.sort()
    return [[round(pts[i][0],5),round(pts[i][1],5)] for i in keep]

def  TryParseEncodedValue(value):	#from https://msdn.microsoft.com/en-us/library/dn306801.aspx
	
	safeCharacters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-";

	parsedValue = list()
	l = list()
	index = 0
	xsum = 0
	ysum = 0

	while (index < len(value)):		# While we have more data,
	
		n = 0				 # initialize the accumulator
		k = 0					  # initialize the count of bits

		while True:
		
			if index >= len(value):  # If we ran out of data mid-number
				return False			# indicate failure.

			b = safeCharacters.find(value[index])
			index += 1
			if b == -1:				# If the character wasn't on the valid list,
				return False			# indicate failure.

			n = n | ((long(b) & 31) << k)	# mask off the top bit and append the rest to the accumulator
			k += 5					 # move to the next position
			# print b
			if b < 32: break		  # If the top bit was not set, we're done with this number.
		

		# The resulting number encodes an x, y pair in the following way:
		#
		#  ^ Y
		#  |
		#  14
		#  9 13
		#  5 8 12
		#  2 4 7 11
		#  0 1 3 6 10 ---> X

		# determine which diagonal it's on
		diagonal = (int)((math.sqrt(8 * n + 5) - 1) / 2)

		# subtract the total number of points from lower diagonals
		n -= diagonal * (diagonal + 1L) / 2

		# get the X and Y from what's left over
		ny = int(n)
		nx = diagonal - ny

		# undo the sign encoding
		nx = (nx >> 1) ^ -(nx & 1)
		ny = (ny >> 1) ^ -(ny & 1)

		# undo the delta encoding
		xsum += nx
		ysum += ny

		# position the decimal point
		l.append((ysum * 0.00001, xsum * 0.00001))
		
	return l



BingMapsKey = 'Ah9Jz-lSTQDUSw5wkstdWU7WOpLPlcWwJ808qb_JYyB7lUjncM12qUJ1xXQQbC2O'


# for i in range(chunkCt):
filename_trip = 'trip_data_' + str(i) + '.csv'
filename_fare = 'trip_fare_' + str(i) + '.csv'
# filename_loc = 'zipcode-location.csv'
mapPoly = {}
# ziploc = {}
disloc = {}


# print disloc
disloc = {}
disloc['Staten_island'] = '40.598601,-74.149969'
disloc['Brooklyn'] = '40.627037,-73.981010'
disloc['Queens'] = '40.751839,-73.814365'
disloc['Manhattan'] = '40.743383,-73.989294'
disloc['Bronx'] = '40.863735,-73.865653'


for k,v in disloc.iteritems():
	url = "https://platform.bing.com/geo/spatial/v1/public/geodata?spatialFilter=GetBoundary(" + v + ",0,'PostCode2',0,0,'en','us')&$format=json&key=" + BingMapsKey
	# print url
	try:
		response = urllib2.urlopen(url)
		data = json.load(response)
	except urllib2.URLError, e:
		print e.code
		continue 
	try:
		s = data['d']['results'][0]['Primitives'][0]['Shape'][2:]
	except IndexError:
		continue
	# print s
	l = list()
	l = TryParseEncodedValue(s)
	if l:
		l = simplify_points(l,0.0005)
		if not (k in mapPoly):
			mapPoly[k] =  [(x[1],x[0]) for x in l]

print mapPoly
with open('polypath.csv','wb') as output:
	writer = csv.writer(output,decimal = ',')
	writer.writerows(mapPoly.values())
	writer.close()

# patchs = []
# fig = plt.figure() 
# ax = fig.gca()
# for k,p in mapPoly.iteritems():
# 	path = Path(p)
# 	patch = patches.PathPatch(path)
# 	patchs.append(patch)
# 	# ax.add_patch(patch)
# collection = PatchCollection(patchs, cmap=plt.cm.hsv, alpha=0.3)
# ax.add_collection(collection)
# ax.axis('scaled')
# ax.axes.get_xaxis().set_ticks([])
# ax.axes.get_yaxis().set_ticks([])

# # plt.show()
# plt.savefig('nymap',ext='png')
# # 	plt.savefig(k,ext = 'png')
