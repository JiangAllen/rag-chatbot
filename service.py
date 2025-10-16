from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureKeyCredential
import pandas as pd
import config
import time
import os
import re

class azure_service:
    def __init__(self, openai_key=None, search_key=None, blob_key=None, openai_resource=None, search_service=None, storage_account=None, api_version=None, index_name=None, container=None):
        self.OPENAI_KEY = openai_key if openai_key is not None else config.openai_key
        self.SEARCHKEY = search_key if search_key is not None else config.search_key
        self.BLOBKEY = blob_key if blob_key is not None else config.blob_key
        self.OPENAI_RESOURCE = openai_resource if openai_resource is not None else config.openai_resource
        self.SEARCH_SERVICE = search_service if search_service is not None else config.search_service
        self.STORAGE_ACCOUNT = storage_account if storage_account is not None else config.storage_account
        self.API_VERSION = api_version if api_version is not None else config.api_version
        self.INDEX_NAME = index_name if index_name is not None else config.index_name
        self.CONTAINER = container if container is not None else config.blob_container
        self.openai_client = AzureOpenAI(
            api_key=self.OPENAI_KEY,
            api_version=self.API_VERSION,
            azure_endpoint="https://{}.openai.azure.com".format(self.OPENAI_RESOURCE)
        )
        self.search_client = SearchClient(
            endpoint="https://{}.search.windows.net".format(self.SEARCH_SERVICE),
            index_name=self.INDEX_NAME,
            credential=AzureKeyCredential(self.SEARCHKEY)
        )
        self.index_client = SearchIndexClient(
            endpoint="https://{}.search.windows.net/".format(self.SEARCH_SERVICE),
            credential=AzureKeyCredential(self.SEARCHKEY)
        )
        self.blob_client = BlobServiceClient(
            account_url="https://{}.blob.core.windows.net".format(self.STORAGE_ACCOUNT),
            credential=self.BLOBKEY
        )
        self.blob_container = self.blob_client.get_container_client(self.CONTAINER)

    def cognitive_search(self, query_text, filters, top):
        r = self.search_client.search(
            query_text,
            filter=filters,
            query_type=QueryType.SEMANTIC,
            query_language="en-us",
            query_speller="lexicon",
            semantic_configuration_name="complex",
            top=top,
            highlight_fields="body",
            highlight_pre_tag="<strong>",
            highlight_post_tag="</strong>"
        )
        r = pd.DataFrame(r)
        return r

    def upload_blobs(self, filename):
        self.blob_client.timeout = 300
        if not self.blob_container.exists():
            self.blob_container.create_container()
        blob_name = os.path.basename(filename).lower()
        with open(filename, "rb") as data:
            self.blob_container.upload_blob(blob_name, data, overwrite=True)

    def blob_name_from_file_page(self, filename):
        return os.path.basename(filename)

    def create_search_index(self, search_index):
        if self.index_client not in self.index_client.list_index_names():
            index = SearchIndex(
                name=search_index,
                fields=[
                    SimpleField(name="news_key", type="Edm.String", key=True),
                    SearchableField(name="subject", type="Edm.String", analyzer_name="zh-Hant.lucene"),
                    SimpleField(name="datepublish", type="Edm.DateTimeOffset", filterable=True, facetable=True),
                    SearchableField(name="keyword", type="Edm.String", analyzer_name="zh-Hant.lucene"),
                    SearchableField(name="body", type="Edm.String", analyzer_name="zh-Hant.lucene"),
                    SearchableField(name="reporter", type="Edm.String", analyzer_name="zh-Hant.lucene"),
                    SimpleField(name="sourcetype", type="Edm.String", filterable=True, facetable=True),
                    SimpleField(name="sourcefile", type="Edm.String", filterable=True, facetable=True),
                    SimpleField(name="cat1", type="Edm.String", filterable=True, facetable=True),
                    SimpleField(name="cat2", type="Edm.String", filterable=True, facetable=True),
                    SimpleField(name="cat3", type="Edm.String", filterable=True, facetable=True)
                ],
                semantic_settings=SemanticSettings(
                    configurations=[SemanticConfiguration(
                        name="complex",
                        prioritized_fields=PrioritizedFields(
                            title_field=None,
                            prioritized_content_fields=[
                                SemanticField(field_name="subject"),
                                SemanticField(field_name="keyword"),
                                SemanticField(field_name="body"),
                                SemanticField(field_name="reporter"),
                                SemanticField(field_name="sourcetype")
                            ]
                        )
                    )]
                )
            )
            self.index_client.create_index(index)

    def index_sections(self, sections):
        i = 0
        batch = []
        for s in sections:
            batch.append(s)
            i += 1
            if i % 1000 == 0:
                results = self.search_client.upload_documents(documents=batch)
                succeeded = sum([1 for r in results if r.succeeded])
                batch = []
        if len(batch) > 0:
            results = self.search_client.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])

    def remove_blobs(self, filename):
        if self.blob_container.exists():
            if filename == None:
                blobs = self.blob_container.list_blob_names()
            else:
                prefix = os.path.splitext(os.path.basename(filename).lower())[0]
                blobs = filter(lambda b: re.match(f"{prefix}.txt", b), self.blob_container.list_blob_names(name_starts_with=os.path.splitext(os.path.basename(prefix))[0]))
            for b in blobs:
                self.blob_container.delete_blob(b)

    def remove_from_index(self, filename):
        while True:
            filter = None if filename == None else f"sourcefile eq '{os.path.basename(filename).lower()}'"
            r = self.search_client.search("", filter=filter, top=1000, include_total_count=True)
            if r.get_count() == 0:
                break
            r = self.search_client.delete_documents(documents=[{"news_key": d["news_key"]} for d in r])
            time.sleep(2)