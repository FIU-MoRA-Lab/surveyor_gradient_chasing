from datetime import datetime, timezone
from collections import deque
import pymongo
import time
import threading
from surveyor_library.surveyor_lib.clients.exo2_client import PARAMS_DICT

class MongoDBDataSender:
    PARAMS_DICT_REVERSED = {v: str(k) for k, v in PARAMS_DICT.items()}

    def __init__(self, prefix=None, mission_postfix='',
                 db_name="missions", queue_maxlen=500, sleep_time=0.2):
        self.api_key = (
            " "
        )
        self.client = pymongo.MongoClient(self.api_key)
        self.db = self.client[db_name]
        self.data_queue = deque(maxlen=queue_maxlen)
        prefix = prefix or datetime.now().strftime("%Y%m%d%H%M")
        self.collection_name = f"{prefix}_{mission_postfix}"
        self.sleep_time = sleep_time
        self._is_thread_alive = True
        self.sending_thread = self.initialize_sender()

        print(f"Sending data to MongoDB collection: {self.collection_name}")

    def initialize_sender(self):
        """
        Initialize the MongoDB collection and start the sender thread.
        """
        self.collection = self.db[self.collection_name]
        print(f"Initialized MongoDB collection: {self.collection_name}")
        return threading.Thread(target=self.queue_sender_to_mongo, daemon=True).start()
        

    def queue_sender_to_mongo(self):
        """
        Continuously send data from a queue to MongoDB.
        """
        while self._is_thread_alive:
            if not self.data_queue:
                time.sleep(self.sleep_time)  # Adjust sleep time as needed
            else:
                data = self.data_queue.popleft()
                try:
                    self.collection.insert_one(data)
                except pymongo.errors.PyMongoError as e:
                    print(f"Error sending data to MongoDB: {e}")
                    self.data_queue.appendleft(data)  # Re-queue the data
                    time.sleep(2.)  # Wait before retrying

            

    def send_to_mongo(self, exodata, metadata):
        """
        Send boat and exo2 data to MongoDB.
        """
        data = {
            "date": metadata.pop("Date"),
            "time": metadata.pop("Time"),
            "longitude": metadata.pop("Longitude"),
            "latitude": metadata.pop("Latitude"),
            "timestamp": datetime.now(timezone.utc),
            "exodata": {self.PARAMS_DICT_REVERSED[k]: v for k, v in exodata.items()},
            "metadata": metadata,
        }

        self.data_queue.append(data)
        print('queue', len(self.data_queue))
    def close(self):
        """
        Clean up resources before exit.
        """
        self.client.close()
        print(f"Closed MongoDB connection for collection: {self.collection_name}")

    def __del__(self):
        self._is_thread_alive = False
        self.sending_thread.join()
        self.close()
            