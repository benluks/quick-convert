This code is on a branch dedicated to:
1. moving the base-dataset class from audio-file-centric to utterance-centric. 
2. Integrating richer support for csv-based data formatting, meaning a flat-list of rows indicates all information necessary for training, meaning:
    - Folder structures need not be crawled on every instantiation of the dataset
    - Complex training involving many sources of pre-computed features can be consolidated into this flat list
3. However, the existing infrastructure can be used to build scripts that create these csvs, as they already include rich support for files.