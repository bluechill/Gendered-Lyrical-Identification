ZONE_LENGTH_IN_DEGREES = 10

def LatLongToZoneNum(latitude, longitude):
   """Returns a zone number between 0 and 647, based on the given lat/long values.
      Zones are 10 degrees latitude by 10 degrees longitude chunks of area on the globe.
      Latitude and longitude must be double values.
      Latitude values must be between -90 and 90, exclusive.
      Longitude values must be between -180 and 180, exclusive.
      (+,+) for NE, (+,-) for SE, (-,+) for NW, (-,-) for SW."""
      
   if (latitude <= -90 or latitude >= 90):
      raise ValueError("Invalid latitude value of " + str(latitude) + ". (-90,90)")
      
   if (longitude <= -180 or longitude >= 180):
      raise ValueError("Invalid longitude value of " + str(longitude) + ". (-180,180)")
   
   # Get lat long into positive values, then scaled down into zone size
   zone_lat_idx = int((latitude + 90) / ZONE_LENGTH_IN_DEGREES)
   zone_long_idx = int((longitude + 180) / ZONE_LENGTH_IN_DEGREES)
   
   # Calculate zone number
   zone = (360/ZONE_LENGTH_IN_DEGREES)*zone_lat_idx + zone_long_idx
   return zone


def ZoneNumToLatLongBoundary(zone_num):
   """Returns 2 strings, stating the latitude and longitude zone boundaries for the given zone_num.
      zone_num must be between 0 and 647, inclusive."""
      
   if (zone_num < 0 or zone_num > 647):
      raise ValueError("Invalid longitude value of " + str(zone_num) + ". [0, 647]")

   zone_lat_idx = zone_num / (360/ZONE_LENGTH_IN_DEGREES)
   zone_long_idx = zone_num % (360/ZONE_LENGTH_IN_DEGREES)
   
   min_lat = zone_lat_idx*ZONE_LENGTH_IN_DEGREES - 90
   min_long = zone_long_idx*ZONE_LENGTH_IN_DEGREES - 180
   
   lat_info = "Latitude range: " + str(min_lat) + " to " + str(min_lat + ZONE_LENGTH_IN_DEGREES)
   long_info = "Longitude range: " + str(min_long) + " to " + str(min_long + ZONE_LENGTH_IN_DEGREES)
   
   return lat_info, long_info;
