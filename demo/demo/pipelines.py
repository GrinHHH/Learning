# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import pymongo
from scrapy import Item


class DemoPipeline(object):
    MONGODB_SERVER = "localhost"
    MONGODB_PORT = 27017
    MONGODB_NAME = 'bgm'
    MONGODB_COL = 'topic-boring-6.25'

    def __init__(self):
        # self.f = open('data.txt','a')
        # myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        # mydb = myclient["bgm"]
        # mycol = mydb["topic-boring"]
        try:
            self.client = pymongo.MongoClient(self.MONGODB_SERVER, self.MONGODB_PORT)
            self.db = self.client[self.MONGODB_NAME]
            self.col = self.db[self.MONGODB_COL]
        except Exception as e:
            print("ERROR): %s" % (str(e),))

    def process_item(self, item, spider):

        try:
            data = dict(item) if isinstance(item, Item) else item
            self.col.insert_one(data)
        except Exception as e:
            print('Error:%s'%e)
        # pos = 0
        # for key,value in item.items():
        #     if pos != 3:
        #         self.f.write(key+' '+value+' ')
        #     else:
        #         self.f.write(key+' '+value+'\n')
        #     pos += 1
        return item



