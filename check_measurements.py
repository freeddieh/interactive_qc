import numpy as np
import pandas as pd
from typing import Any, Tuple, List


class Observation:
    def __init__(self, 
                 name: str, 
                 measurements: str | List[Any, float | np.float64] | np.ndarray[Any, float | np.float64] | pd.DataFrame, 
                 unit: str) -> None:
        
        self.name = name
        self.unit = unit

        # Find and format the measurements as a Pandas DataFrame if not already of that format
        if type(measurements) == str:
            measurements_placeholder = pd.read_csv(measurements)
        elif type(measurements) == np.ndarray:
            measurements_placeholder = pd.DataFrame(measurements)
        elif type(measurements) == list:
            measurements_placeholder =  pd.DataFrame(measurements)
        else:
            raise ValueError('The format of the measurements are not of an accepted format (list, np.ndarray or pd.DataFrame).')
        
        # Find the Date column of the DataFrame
        try:
            measurements_placeholder.iloc[:, 0].astype(float)
            measurements_placeholder.columns.values[0] = f'{name} [{unit}]'
            measurements_placeholder.columns.values[1] = 'Date'
        except ValueError:
            measurements_placeholder.columns.values[0] = 'Date'
            measurements_placeholder.columns.values[1] = f'{name} [{unit}]'

        self.measurements = measurements_placeholder


    def __str__(self) -> str:
        return f'This is measurements of {self.name} with {len(self.measurements)} measurements.'


class DatabaseObservation(Observation):
    def __init__(self, 
                 name: str, 
                 measurements: str | List[Any, float | np.float64] | np.ndarray[Any, float | np.float64] | pd.DataFrame, 
                 unit: str) -> None:
        super().__init__(name, measurements, unit)

    
    def compare_to_original(self, original: 'Observation') -> None:
        """
        
        """
        if self.name != original.name:
            raise TypeError('The samples to be compared are of two different species.')
        elif self.unit != original.unit:
            raise ValueError('The units of these observations are different from the original.')
        elif self.measurements['Date'] != original.measurements['Date']:
            raise ValueError('The the observations are of two different timeperiods.')
        else:
            if self.measurements[f'{self.name} [{self.unit}]'] != original.measurements[f'{original.name} [{original.unit}]']:
                print('The two observations are identical.')


if __name__ == '__main__':
    species_name, species_unit = str(input('Please input name and unit of the species, separated by ",":')).split(',')
    original_measurements = str(input('Please input path to the original measurements:')) 
    database_measurements = str(input('Please input path to the database measurements:'))

    Obs = Observation(name=species_name, unit=species_unit, measurements=original_measurements)
    DBobs = DatabaseObservation(name=species_name, unit=species_unit, measurements=database_measurements)

    DBobs.compare_to_original(original=Obs)
