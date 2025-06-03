from datetime import datetime, timezone

import pymongo

from surveyor_library.surveyor_lib.clients.exo2_client import PARAMS_DICT

# Reverse PARAMS_DICT for easier lookup
PARAMS_DICT_REVERSED = {v: str(k) for k, v in PARAMS_DICT.items()}

# MongoDB connection setup
API_KEY = (
    "your apikey"  # Replace with your MongoDB connection string
)
client = pymongo.MongoClient(API_KEY)
db = client["missions"]


def send_to_mongo(boat, asvid, mission_postfix=""):
    """
    Send boat and exo2 data to MongoDB.

    Args:
        boat: Boat object with get_data method.
        asvid: ASV identifier.
        mission_postfix: Optional postfix for collection name.
    """
    postfix = send_to_mongo.postfix
    collection_name = f"{postfix}"
    if mission_postfix:
        collection_name += f"_{mission_postfix}"
    collection = db[collection_name]

    # Gather data
    exo_data = boat.get_data(["exo2"])
    boat_data = boat.get_data(["state"])
    print(exo_data, '------', boat_data)

    data = {
        "date": boat_data.pop("Date"),
        "time": boat_data.pop("Time"),
        "longitude": boat_data.pop("Longitude"),
        "latitude": boat_data.pop("Latitude"),
        "timestamp": datetime.now(timezone.utc),
        "exodata": {PARAMS_DICT_REVERSED[k]: v for k, v in exo_data.items()},
        "metadata": {**boat_data, "asvid": asvid},
    }

    collection.insert_one(data)
    # print(f"Data sent to MongoDB successfully to missions/{collection_name}.")


# Default postfix for mission data
send_to_mongo.postfix = datetime.now().strftime("%Y%m%d%H%M%S")
