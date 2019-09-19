import json
import os
import re
import shutil
import tarfile
import numpy as np
import pandas as pd
import requests
import h5py
from tqdm.autonotebook import tqdm
from pathlib import Path

Path.ls = lambda x: [o.name for o in x.iterdir()]

def unzipFile(fname, path="."):
    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            # Extract member
            tar.extract(member=member, path=path)
        tar.extractall(path=path)
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            # Extract member
            tar.extract(member=member, path=path)
        # tar.extractall(path=path)
        tar.close()


def extractFilesFromDirs(dir_list, path="."):
    # extract downloaded and unzipped files from directories
    # in path, whose name is in dir_list
    for dir in dir_list:
        filepath = path + "/" + dir
        # print(filepath)
        for root, dirs, files in os.walk(filepath, topdown=False):
            for file in files:
                try:
                    # print(file)
                    shutil.move(filepath + "/" + file, path + "/" + file)
                except OSError:
                    pass
                # delete directories
                shutil.rmtree(filepath)


def getDataEx(disease_types=["Squamous Cell Neoplasms", "Adenomas and Adenocarcinomas"], path=".", file_n=2):
    # Query example:
    # cases.disease_type in ["Adenomas and Adenocarcinomas","Squamous Cell Neoplasms"]
    # and cases.primary_site in ["Bronchus and lung"]
    # and files.data_category in ["DNA Methylation"]
    # and files.platform in ["Illumina Human Methylation 450"]
    # and cases.samples.sample_type in ["Primary Tumor","Recurrent Tumor"]

    pathh = os.path.join(os.getcwd(), path)
    if not os.path.isdir(pathh):
        os.makedirs(pathh)

    files_endpt = "https://api.gdc.cancer.gov/files"

    # create one folter for each cancer types (if there are not)
    for disease_type in disease_types:
        # create category folder (every category must have their folder)
        outputDir = os.path.join(pathh, disease_type)
        if not os.path.isdir(outputDir):
            os.mkdir(outputDir)

        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.disease_type",
                        "value": disease_type
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "cases.primary_site",
                        "value": ["Bronchus and lung"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_category",
                        "value": ["DNA Methylation"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.platform",
                        "value": ["Illumina Human Methylation 450"]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "cases.samples.sample_type",
                        "value": ["Primary Tumor", "Recurrent Tumor"]
                    }
                }
            ]
        }

        fields = ["file_id", "file_name"]
        fields = ",".join(fields)

        # A POST is used, so the filter parameters can be passed directly as a Dict object.
        params = {
            "filters": filters,
            "fields": fields,
            "format": "JSON",
            "size": file_n
        }

        # The parameters are passed to 'json' rather than 'params' in this case
        response = requests.post(files_endpt,
                                 headers={"Content-Type": "application/json"},
                                 json=params,
                                 stream=True)
        # print(response.content.decode("utf-8"))

        file_list = []

        # This step populates the download list with the file_ids from the previous query
        # JSON format is { "data": { "hits": ["file_id":"...", "id":"..." ] } }
        # for file_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
        #    file_uuid_list.append(file_entry["file_id"])
        file_list = json.loads(response.content.decode("utf-8"))["data"]["hits"]

        # print("list of files to be downloaded:\n",file_uuid_list)
        print("Files belonging to %s: %d" % (disease_type, len(file_list)))

        # before downloading data, check which files are already been
        # downloaded (thus list files inside path directory)
        file_already_downloaded = []
        file_tobe_downloaded = []

        for root, dirs, files in os.walk(outputDir, topdown=False):
            file_already_downloaded.extend(files)

        print("Files already downloaded: %d" % (len(file_already_downloaded)))

        # filter return an iterable: thus i have to cast to list
        # file_tobe_downloaded = list(filter(lambda x: x['file_name'] not in file_already_downloaded, file_list))
        for x in file_list:
            if x['file_name'] not in file_already_downloaded:
                file_tobe_downloaded.append(x['file_id'])

        # download only data not already downloaded
        print("Files to be downloaded: %d" % (len(file_tobe_downloaded)))

        data_endpt = "https://api.gdc.cancer.gov/data"

        params = {"ids": file_tobe_downloaded}

        print("download in progress...")
        # query data corresponding to file ids
        response = requests.post(data_endpt,
                                 data=json.dumps(params),
                                 headers={"Content-Type": "application/json"},
                                 stream=True
                                 )

        response_head_cd = response.headers["Content-Disposition"]
        # print("resp headers", response.headers)
        file_name = outputDir + "/" + re.findall("filename=(.+)", response_head_cd)[0]
        print("filename ", file_name)
        total_size = int(40 * 1024 * 1024 * len(file_tobe_downloaded))
        block_size = 1024 * 1024
        print("File size: {}MB".format(total_size / block_size))
        wrote = 0
        progress = tqdm(unit="B", unit_scale=True, total=total_size, unit_divisor=1024)
        with open(file_name, "wb") as output_file:
            for data in response.iter_content(chunk_size=block_size):
                if data:
                    wrote = wrote + len(data)
                    progress.update(len(data))
                    output_file.write(data)
                    # output_file.write(response.content)
        progress.close()
        
        print("data downloaded")
        
        unzipFileEx(file_name)
        
        extractFilesFromDirsEx(outputDir)
        


def getSaneDataEx(path=".", file_n=71):
    #path = Path("./data/sane")
    pathh = os.path.join(os.getcwd(), path)
    if not os.path.isdir(pathh):
        os.makedirs(pathh)

    outputDir = os.path.join(pathh, 'sane')
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    files_endpt = "https://api.gdc.cancer.gov/files"

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.disease_type",
                    "value": ["Squamous Cell Neoplasms", "Adenomas and Adenocarcinomas"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.primary_site",
                    "value": ["Bronchus and lung"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_category",
                    "value": ["DNA Methylation"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "files.platform",
                    "value": ["Illumina Human Methylation 450"]
                }
            },
            {
                "op": "in",
                "content": {
                    "field": "cases.samples.sample_type",
                    "value": ["Solid Tissue Normal"]
                }
            }
        ]
    }

    fields = ["file_id", "file_name"]
    fields = ",".join(fields)

    # A POST is used, so the filter parameters can be passed directly as a Dict object.
    params = {
        "filters": filters,
        "fields": fields,
        "format": "JSON",
        "size": file_n
    }

    # The parameters are passed to 'json' rather than 'params' in this case
    response = requests.post(files_endpt,
                             headers={"Content-Type": "application/json"},
                             json=params,
                             stream=True)
    # print(response.content.decode("utf-8"))

    file_list = []

    # This step populates the download list with the file_ids from the previous query
    # JSON format is { "data": { "hits": ["file_id":"...", "id":"..." ] } }
    # for file_entry in json.loads(response.content.decode("utf-8"))["data"]["hits"]:
    #    file_uuid_list.append(file_entry["file_id"])
    file_list = json.loads(response.content.decode("utf-8"))["data"]["hits"]

    # print("list of files to be downloaded:\n",file_uuid_list)
    # print("Files belonging to %s: %d" %(disease_type,len(file_list)))

    # before downloading data, check which files are already been
    # downloaded (thus list files inside path directory)
    file_already_downloaded = []
    file_tobe_downloaded = []

    for root, dirs, files in os.walk(outputDir, topdown=False):
        file_already_downloaded.extend(files)

    print("Files already downloaded: %d" % (len(file_already_downloaded)))

    # filter return an iterable: thus i have to cast to list
    # file_tobe_downloaded = list(filter(lambda x: x['file_name'] not in file_already_downloaded, file_list))
    for x in file_list:
        if x['file_name'] not in file_already_downloaded:
            file_tobe_downloaded.append(x['file_id'])

    # download only data not already downloaded
    print("Files to be downloaded: %d" % (len(file_tobe_downloaded)))

    data_endpt = "https://api.gdc.cancer.gov/data"

    params = {"ids": file_tobe_downloaded}

    print("download in progress...")
    # query data corresponding to file ids
    response = requests.post(data_endpt,
                             data=json.dumps(params),
                             headers={"Content-Type": "application/json"},
                             stream=True
                             )

    response_head_cd = response.headers["Content-Disposition"]
    # print("resp headers", response.headers)
    file_name = outputDir + "/" + re.findall("filename=(.+)", response_head_cd)[0]
    # import pdb; pdb.set_trace()
    print("filename ", file_name)
    total_size = int(40 * 1024 * 1024 * len(file_tobe_downloaded))
    block_size = 1024 * 1024
    print("File size: {}MB".format(total_size / block_size))
    wrote = 0
    progress = tqdm(unit="B", unit_scale=True, total=total_size, unit_divisor=1024)
    with open(file_name, "wb") as output_file:
        for data in response.iter_content(chunk_size=block_size):
            if data:
                wrote = wrote + len(data)
                progress.update(len(data))
                output_file.write(data)
                # output_file.write(response.content)
    progress.close()
    print("data downloaded")
    
    unzipFileEx(file_name)
    
    extractFilesFromDirsEx(outputDir)


def extractFilesFromDirsEx(path="."):
    # extract files from each dirs contained
    # in path
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            filepath = path + "/" + dir
            # print(filepath)
            for root, dirs, files in os.walk(filepath, topdown=False):
                for file in files:
                    try:
                        # print(file)
                        shutil.move(filepath + "/" + file, path + "/" + file)
                    except OSError:
                        pass
                    # delete directories
                    shutil.rmtree(filepath)


# unzip file fname and then removes it
def unzipFileEx(fname):
    path = os.path.dirname(fname)
    # print("unzip %s -> in %s" %(fname,path))

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            # Extract member
            tar.extract(member=member, path=path)
        tar.extractall(path=path)
        tar.close()
    elif fname.endswith("tar"):
        tar = tarfile.open(fname, "r:")
        for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
            # Extract member
            tar.extract(member=member, path=path)
        # tar.extractall(path=path)
        tar.close()


def storeDataIntoBinary(path="./data"):

    disease_types = ["Adenomas and Adenocarcinomas", "Squamous Cell Neoplasms", "sane"]

    column_names = ["Composite", "Beta_value", "Chromosome", "Start", "End", "Gene_Symbol", "Gene_Type",
                    "Transcript_ID", "Position_to_TSS", "CGI_Coordinate", "Feature_Type"]
    chrms = ['chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19',
             'chr2', 'chr20', 'chr21', 'chr22', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX',
             'chrY']

    chrms_position = dict()

    cpgs_nan = None

    for idx, disease_type in enumerate(disease_types):
        # distributions[disease_type] = dict()

        disease_type_path = Path(path) / disease_type
        if not disease_type_path.is_dir():
            continue
        files = [f for f in os.listdir(disease_type_path) if os.path.isfile(os.path.join(disease_type_path, f))]
        na_vals = "."
        fillna = -1

        all_betas = None
        for file in tqdm(files):
            # not considering the MANIFEST and the zip
        
            if file != "MANIFEST.txt" and not file.endswith(".tar") and not file.endswith("tar.gz"):
                # crea DataFrame
                dft_tmp = pd.read_csv(os.path.join(disease_type_path, file), sep="\t", header=None,
                                      usecols=['Composite', 'Beta_value', 'Chromosome'], names=column_names, skiprows=1,
                                      na_values=na_vals)  # ,keep_default_na=False)
                patternDel = "^cg"
                filter = dft_tmp['Composite'].str.contains(patternDel)
                dft_tmp = dft_tmp[filter]
                dft_tmp.Beta_value = dft_tmp.Beta_value.astype('float64')
                dft_tmp = dft_tmp[dft_tmp.Chromosome != '*']

                if cpgs_nan is None:
                    cpgs_nan = dft_tmp.Beta_value.isnull()
                else:
                    cpgs_nan = np.logical_or(cpgs_nan, dft_tmp.Beta_value.isnull())

                betas = np.zeros(0)
                last = 0
                for cromosoma in chrms:
                    dft_chm = dft_tmp.loc[dft_tmp.Chromosome == cromosoma]
                    if fillna == -1:
                        # calcola mean value del beta_value su quel cromosoma
                        mean_value = dft_chm['Beta_value'].mean(axis=0)
                        # sostituisci i NaN con il mean value
                        dft_chm['Beta_value'].fillna(mean_value, inplace=True)
                    else:
                        mean_value = fillna
                    dft_chm['Beta_value'].fillna(mean_value, inplace=True)
                    tmp = np.asarray(dft_chm.Beta_value.tolist())
                    del dft_chm
                    betas = np.concatenate((betas, tmp))
                    if cromosoma not in chrms_position:
                        end = len(betas)
                        chrms_position[cromosoma] = (last, end)
                        last = end
                if all_betas is None:
                    all_betas = np.expand_dims(betas, 0)
                else:
                    all_betas = np.concatenate((all_betas, np.expand_dims(betas, 0)))
        # end = len(all_betas)
        # disease_position[disease_type] = (init_d, end)
        # init_d = end
        with h5py.File('data/data.h5', 'w' if idx == 0 else 'a') as hf:
            hf.create_dataset(disease_type, data=all_betas)

    print("Saving chrms pos...")
    # # compressed = zlib.compress(pickle.dumps(distributions), zlib.Z_BEST_COMPRESSION)
    # compressed = zlib.compress(str.encode(json.dumps(distributions)))
    with open('./data/chrms.dat', 'wb') as f:
        j = json.dumps(chrms_position)
        f.write(str.encode(j))
    with h5py.File('data/cpgs_nan.h5', 'w') as hf:
        hf.create_dataset("cpgs_nan", data=cpgs_nan)
    # f.write(compressed)
    # f.close()
    print("Saved")
