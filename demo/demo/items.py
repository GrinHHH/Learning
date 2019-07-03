# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class DemoItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    topic_name = scrapy.Field()
    author = scrapy.Field()
    post_num = scrapy.Field()
    last_post = scrapy.Field()
    pass
