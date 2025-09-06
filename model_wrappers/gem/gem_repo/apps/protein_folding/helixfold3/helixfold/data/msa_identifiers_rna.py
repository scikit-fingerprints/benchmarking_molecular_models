#   Copyright (c) 2024 PaddleHelix Authors. All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for extracting identifiers from RNA MSA sequence descriptions."""

from helixfold.data.msa_identifiers import Identifiers


def get_identifiers(description: str, species_identifer_df) -> Identifiers:
  """Computes extra MSA features from the description."""

  identifer = ''

  if species_identifer_df is None:
    return Identifiers(
          species_id=identifer)

  word_list = description.split()
  for i, word in enumerate(word_list):

    if word[0].isupper() and i+1 < len(word_list):
      name =  ' '.join(word_list[i:i+2])

      matching_rows = species_identifer_df.loc[species_identifer_df['Scientific name'] == name, 'Mnemonic']  
      if not matching_rows.empty:  
        identifer = matching_rows.iloc[0]

        break

  return Identifiers(
        species_id=identifer)


