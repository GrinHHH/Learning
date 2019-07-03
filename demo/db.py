import pymongo
import fake_useragent

MONGODB_SERVER = "localhost"
MONGODB_PORT = 27017
MONGODB_NAME = 'bgm'
MONGODB_COL = 'topic-boring-6.25'

client = pymongo.MongoClient(MONGODB_SERVER, MONGODB_PORT)
db = client[MONGODB_NAME]
col = db[MONGODB_COL]
count = 0
for x in col.find():
    if count<10:
        print (x)
    count += 1

