#!/urs/bin/env python
# -*- coding:utf-8 -*-
"""

:Author:  Houyanlong
:Create:  2023/5/26 15:18
Copyright (c) 2018, Lianjia Group All Rights Reserved.
"""
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from configs.model_config import *
import json

if __name__ == '__main__':
    import pydevd_pycharm
    pydevd_pycharm.settrace('39.155.135.248', port=27777, stdoutToServer=True, stderrToServer=True)
    # embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict["text2vec"],
    #                                    model_kwargs={'device': "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"})
    # vector_store = FAISS.load_local('vector_store/kefu_final', embeddings)
    # res = []
    # with open("content/kefu_final/kefu_langchain_only_query.txt") as f:
    #     _dict = {}
    #     _str = f.readline().replace("#0525#", "")
    #     docs = vector_store.similarity_search_with_score(_str,k=10)
    #     _dict[_str] = [(docs[0].page_content, docs[1]) for _ in docs]
    #     res.append(_dict)
    # with open("kefu_query_embedding_scores.txt","w") as f:
    #     for line in res:
    #         f.write(json.dumps(line))
    #         f.write("\n")

    with open("kefu_query_embedding_scores.txt") as f:
        while True:
            _str = f.readline()
            if not _str:
                break



