# -*- coding: utf-8 -*-
import scrapy
import time
from ..items import DemoItem


class TestSpider(scrapy.Spider):
    name = 'test'
    # allowed_domains = ['bgm.tv']
    start_urls = ['https://bgm.tv/group/boring/forum']

    def parse(self, response):
        contents = response.css('table.topic_list tbody')

        for topic in contents.css('tr'):
            item = DemoItem()
            cut = topic.css('td.lastpost small::text').get().split('-')[0]
            if cut!='2019':
                return None
            else:
                # yield {
                #     'topic_name': topic.css('td.subject a::text').extract_first(),
                #     'author': topic.css('td.author a::text').extract_first(),
                #     'post_num': topic.css('td.posts::text').extract_first(),
                #     'last_post': topic.css('td.lastpost small::text').extract_first()
                # }
                item['author'] = topic.css('td.author a::text').extract_first()
                item['last_post'] = topic.css('td.lastpost small::text').extract_first()
                item['post_num'] = topic.css('td.posts::text').extract_first()
                item['topic_name'] = topic.css('td.subject a::text').extract_first()
                yield item
        print('one done')
        next_url = response.css('div.page_inner a.p::attr(href)')[-2].get()
        if next_url is not None:
            time.sleep(1)
            yield response.follow(next_url,callback=self.parse)