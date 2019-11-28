#-*- coding: utf-8 -*-

import os
import numpy as np
from collections import OrderedDict
from collections import Counter
from math import log
import operator
from pprint import pprint
import pickle


def construct_tf_idf_matrix(Feature_Info, Doc_Term_Count_Dict, is_debug = False):
    TF_Dict = calculate_TF(Feature_Info, Doc_Term_Count_Dict)
    IDF_Dict = calculate_IDF(Feature_Info, Doc_Term_Count_Dict)

    # construct TF_IDF dictionary [ TF_IDF[document_path][term_name] ]
    TF_IDF_Dict = OrderedDict()
    for doc_name in TF_Dict.keys():
        TF_IDF_Dict[doc_name] = OrderedDict()
        for term_name in TF_Dict[doc_name].keys():
            TF_IDF_Dict[doc_name][term_name] = TF_Dict[doc_name][term_name] * IDF_Dict[term_name]

    ''' # Debugging
    print (len(TF_Dict))
    for idx, elem in enumerate(TF_Dict):
        print (elem, len(TF_Dict[elem]))

        if idx>10:  break

    print("PAUSE")
    A = input()
    '''

    # Normalization
    


    return TF_IDF_Dict

def construct_POS_Dictionary(parameters, docpath_list, is_train=False):

    Term_Counter = Counter()

    # Doc_POS_Freq_Dict : 한문서 안에 나타난 명사형 형태소의 빈도수 저장하는 사전
    Doc_POS_Freq_Dict = OrderedDict()  # [Document][Term] Dictionary
    NOUN_detectlist = ['NNG', 'NNP']  # 일반명사, 고유명사

    for cnt, filepath in enumerate(docpath_list):

        # 해당 문서 안에서 형태소의 발생 빈도를 저장하는 사전 구성
        Doc_POS_Freq_Dict[filepath] = OrderedDict()
        with open(filepath, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
            for elem in lines:
                # 문서에서 한줄씩
                if elem == ' ' or elem =='\n' or elem=='\r':
                    # 한문장에 대한 형/분이 끝나면, 빈 줄이 나옴--> 그거 처리
                    continue
                POS_Result = elem.split('\t')[1].strip()    # 한 어절에 대한 형태소 분석 결과
                POS_Result = POS_exception_check(POS_Result)    # 예외처리

                POS_temp_list = POS_Result.split('+')       # 각 element는 형태소 하나씩임(서강대/NNG)
                for elem in POS_temp_list:
                    POS_elem = elem.split('/')
                    if POS_elem[1] in NOUN_detectlist:
                        # elem(서강대/NNG)를 doc_term_count_dict에 추가
                        if elem in Doc_POS_Freq_Dict[filepath].keys():
                            Doc_POS_Freq_Dict[filepath][elem] += 1
                        else:
                            Doc_POS_Freq_Dict[filepath][elem] = 1

                        if is_train == True:
                            Term_Counter.update([elem])


    # Debugging
    if is_train == True:
        print ("")
        print ("-------------- POS Feature ---------------")
        print ("")

        for term_name, term_cnt in Term_Counter.most_common(30):
            print ("{} : {}". format(term_name, term_cnt))

        print ("")
        print ("------------------------------------------")
        print ("")
        print ("Press 123")
        A = input()
    


    # 전체 체언류 형태소 중 빈도수가 높은 ('max_feature_dim')만을 추출
    # 체언류 형태소 개수가 너무 많아서 학습시간이 너무 오래 걸림...
    #전체 문서에 나온 형태소 중 상위 'max_feature_dim'개의 term을 list로 저장
    if is_train == True:
        Term_Selection = []
        for term_name, term_cnt in Term_Counter.most_common(parameters['max_feature_dim']):
            Term_Selection.append(term_name)

        return Term_Selection, Doc_POS_Freq_Dict

    return None, Doc_POS_Freq_Dict



def calculate_TF(Term_Selection, Doc_POS_Freq_Dict):

    # 문서에 Term_Selection에 있는 형태소들이 몇개씩 있는지 저장하는 2차원 사전
    Doc_Term_Count_Dict = OrderedDict()
    for doc_path in Doc_POS_Freq_Dict.keys():
        Doc_Term_Count_Dict[doc_path] = OrderedDict()

        for POS_elem in Term_Selection:
            if POS_elem in Doc_POS_Freq_Dict[doc_path].keys():
                Doc_Term_Count_Dict[doc_path][POS_elem] = Doc_POS_Freq_Dict[doc_path][POS_elem]
            else:
                Doc_Term_Count_Dict[doc_path][POS_elem] = 0

    # Normalization
    for key, value in Doc_Term_Count_Dict.items():
        np_list = np.array(list(value.values()))
        log_np_list = np.log10(np_list+1)

        for norm_elem, value_key_elem in zip(log_np_list, value.keys()):
            value[value_key_elem] = norm_elem

    return Doc_Term_Count_Dict

def calculate_IDF(Term_Selection, Doc_POS_Freq_Dict):
    Term_Doc_Count_Dict = OrderedDict()

    # 하나의 Feature에 대하여
    for feat_elem in Term_Selection:
        # 하나의 문서에서
        for doc_name in Doc_POS_Freq_Dict.keys():
            if feat_elem in Doc_POS_Freq_Dict[doc_name].keys():
                if feat_elem in Term_Doc_Count_Dict.keys():
                    Term_Doc_Count_Dict[feat_elem] += 1

                else:
                    Term_Doc_Count_Dict[feat_elem] = 1

        if feat_elem not in Term_Doc_Count_Dict.keys():
            Term_Doc_Count_Dict[feat_elem] = 0


    Doc_cnt = len(Doc_POS_Freq_Dict)  #1787

    # 각 형태소에 대한 idf값 계산
    for term_name, term_cnt in Term_Doc_Count_Dict.items():
        Term_Doc_Count_Dict[term_name] = log(Doc_cnt / (term_cnt+1))

    return Term_Doc_Count_Dict


def dir_scan(dirname, filescan_list):
    '''
        - 디렉토리 경로를 저장해서 리스트 형태로 리턴
    '''
    try:
        dir_list = os.listdir(dirname)
        for dir_elem in dir_list:
            dir_elem = os.path.join(dirname, dir_elem)

            if os.path.isdir(dir_elem):
                dir_scan(dir_elem, filescan_list)
            else:
                ext = os.path.splitext(dir_elem)[-1]
                if ext == '.txt':
                    #print (dir_elem)
                    filescan_list.append(dir_elem)

    except PermissionError:
        pass

    return filescan_list

def POS_exception_check(POS_info):
    POS_exception_list = ['++/SW', '+//SW', '+/SW+'] #  +/SW+가 ++/SW보다 앞에 있으면 안됨.
    for elem in POS_exception_list:
        POS_info = POS_info.replace(elem, '')
    return POS_info


def main():
    with open("../Data/parameters.bin", "rb") as f:
        parameters = pickle.load(f)

    Tf_Idf_Dict, Term_Selection = calculate_tf_idf(parameters)

    print (len(Tf_Idf_Dict))

    for doc_name in Tf_Idf_Dict.keys():
        tmp_dict = Tf_Idf_Dict[doc_name]
        for idx, elem in enumerate(tmp_dict):
            print (" {} : {:.4f}".format(elem, tmp_dict[elem]))
            if idx > 30: break

        break
        #A = input()
    print ("FINISH")


if __name__ == '__main__':
    main()
