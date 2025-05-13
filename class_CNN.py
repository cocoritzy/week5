{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0f34bb52",
   "metadata": {},
   "outputs": [],
   "source": [
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\") \n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.metrics import confusion_matrix\n",
    "import os, librosa\n",
    "from tqdm import tqdm\n",
    "import torchaudio\n",
    "import IPython.display as ipd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pickle \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e790269b",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('audio_data.pkl', 'rb') as f:\n",
    "    audio_data_df = pickle.load(f)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
