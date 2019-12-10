import osimport math'''searching .txt files' path'''def search(dirname, files):    try:        filenames = os.listdir(dirname)        for filename in filenames:            full_filename = os.path.join(dirname, filename)            if os.path.isdir(full_filename):                search(full_filename, files)            else:                ext = os.path.splitext(full_filename)[-1]                if ext == '.txt':                    files.append(full_filename)    except PermissionError:        pass'''mapping features to indicesreturn : {"갑/NNG":0, "을/NNG":1, ... }'''def idcs_of_features(features_file):    features = {}    with open(features_file, 'r', encoding="utf-8") as f:        i = 0        while True:            line = f.readline()            if not line:                break            key = line.split()[0]            features[key] = i            i += 1    return features'''calculate TF for a document(file)return : [0, 0, 1, ... ]'''def calculate_TF(file, features):    TF = [0.] * len(features)    words = []    with open(file, 'r', encoding='utf-8') as f:        while True:            line = f.readline()            if line is not '\n' and line is not '':                tokens = list(map(str, line.replace(' ', '\t').split('\t')))                if '+' in tokens[1]:                    text = tokens[1].split('+')                    for tx in text:                        words += [tx.replace('\n', '')]                else:                    words += [tokens[1].replace('\n', '')]            if not line:                break    for word in words:        if word in features.keys():            TF[features[word]] += 1    return TF'''calculate IDF for all of documentsreturn : [4, 3, ... ]'''def calculate_IDF(documents, features, TF):    D = len(documents)    IDF = [0] * len(features)    # count IDF    for i in range(D):        # count the IDF using train data set        if 'Input_Data' in documents[i]:            for key in features.keys():                if TF[i][features[key]] > 0:                    IDF[features[key]] += 1    return IDF'''calculate normalized TF-IDF for a filereturn : [0., 0.0033242.., 0., 0., ... ]'''def TF_IDF_Calculation(features, TF, IDF):    TFIDF = [0] * len(features)    TFIDF_sum = 0    for key in features.keys():        TFIDF_sum += (TF[features[key]] * IDF[features[key]]) ** 2    for key in features.keys():        if TFIDF_sum == 0:  # if TF or IDF is zero, then TFIDF_sum is changed to a very small value            TFIDF_sum = 0.00000000000000000000000001        TFIDF[features[key]] = (TF[features[key]] * IDF[features[key]]) / math.sqrt(TFIDF_sum)    return TFIDF'''write TF-IDF values on files with tab'''def make_output_file(output, TFIDF):    writeFile = open(output, 'w', encoding="utf-8")    for (index, value) in enumerate(TFIDF):        if index == 0:            writeFile.write(str(value))        else:            writeFile.write('\t' + str(value))    writeFile.close()    return##def main():    feature_set = "output.txt"  # the result of implement #1    data_path = './Corpus/Input_Data'  # path of Corpus ex ./Corpus    data_path2 = './Corpus/Test_Data'    data_path3 = './Corpus/Val_Data'    paths = []  # paths of Corpus directory    search(data_path, paths)    search(data_path2, paths)    search(data_path3, paths)    write_path = './201735878_정주환/'  # path of write directory    w_paths = []  # paths of result directory    for i in paths:        w_paths.append(i.replace("./Corpus", write_path))    feature_set = idcs_of_features(feature_set)  # indices of each feature    TF = []  # 2D-array for TF values of each documents    for path in paths:  # get the TF values of each feature in documents        TF += [calculate_TF(path, feature_set)]    values = {}    IDF = calculate_IDF(paths, feature_set, TF)  # get IDF values of each feature    for (i, v) in enumerate(feature_set):        values[v[0]] = i    print(IDF)    for (i, path) in enumerate(w_paths):  # get TF-IDF values of each feature in documents        TF_IDF = TF_IDF_Calculation(feature_set, TF[i], IDF)        make_output_file(path, TF_IDF)  # write filesif __name__ == "__main__":    main()