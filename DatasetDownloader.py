import kaggle

kaggle.api.authenticate()

dataset = 'moltean/fruits'
downloaddir = 'data/'

kaggle.api.dataset_download_files(dataset,path=downloaddir, unzip=True)
