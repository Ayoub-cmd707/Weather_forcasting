import sys, os
sys.path.append('/home/jupyter-aaron/Postprocessing/pythie')
import numpy as np
import xarray as xr
import pickle
import gc
import time  # Import time module to measure execution time
from core.data import Data
import postprocessors.MBM as MBM

#THIS IS FOR 100M WINDSPEED
# Parameters and directories
params = ['t2m', 'u10', 'v10', 'tcc', 'u100', 'v100', 'z', 't', 'p10fg6', 'ssrd6', 'strd6']
directory = "../TrainValTestSplit"
base_dir = "../data"
output_dir = './resultsClassicalMBM'  # Directory to save results
time_log_path = os.path.join(output_dir, 'training_times_per_lead_time.txt')
# Function to load forecast data in batches
def load_forecast_data(file_list, base_dir, params, lead):
    print(f"Loading forecast data for lead time {lead}...")
    data_array = np.empty((len(params), len(file_list), 11, 1, 1, 32, 33), dtype=np.float32)
    for i, file in enumerate(file_list):
        file_path = os.path.join(base_dir, file)
        with xr.open_dataset(file_path) as dataset:
            dataset = dataset.isel(step=lead)
            for j, var in enumerate(params):
                data_array[j, i, :, 0, 0, :, :] = dataset[var]
                if np.isnan(data_array[j, i]).any():
                    print(f"⚠️ NaN gevonden in var '{var}' voor bestand: {file}")
   
    print("Forecast data loaded.")
    return data_array

# Function to load observation data in batches
def load_observation_data(file_list, base_dir, lead):
    print(f"Loading observation data for lead time {lead}...")
    data_array = np.empty((1, len(file_list), 1, 1, 1, 32, 33), dtype=np.float32)
    for i, file in enumerate(file_list):
        file_path = os.path.join(base_dir, file)
        with xr.open_dataset(file_path) as dataset:
            dataset = dataset.squeeze("surface")
            dataset = dataset.isel(step=lead)
            data_array[0, i, :, 0, 0, :, :] = dataset['ssrd6_obs']
    print("Observation data loaded.")
    return data_array



# Load testing files (loaded once as these don’t change in the loop)
print("Loading testing forecast and observation data files...")
with open(os.path.join(directory, "test_eupp_files.pkl"), 'rb') as f:
    rfcs_test_file = pickle.load(f)
    #print(rfcs_test_file)
with open(os.path.join(directory, "test_era5_files.pkl"), 'rb') as f:
    obs_test_file = pickle.load(f)
    #print(obs_test_file)


with open(time_log_path, 'w') as time_log_file:
    for lead_time in range(0, 1):
        try:
            print(f"\nProcessing lead time {lead_time}...")


            start_time = time.time()

            print("Loading training forecast data files...")
            with open(os.path.join(directory, "train_eupp_files.pkl"), 'rb') as f:
                rfcs_train_file = pickle.load(f)
        
            rfcs_array_train = load_forecast_data(rfcs_train_file, base_dir, params, lead_time)
            print("NaN check - rfcs_array_train:", np.isnan(rfcs_array_train).any())
            gc.collect()  # Free memory

            print("Loading training observation data files...")
            with open(os.path.join(directory, "train_era5_files.pkl"), 'rb') as f:
                obs_train_file = pickle.load(f)
                
            obs_array_train = load_observation_data(obs_train_file, base_dir, lead_time)
            print("NaN check - obs_array_train:", np.isnan(obs_array_train).any())
            gc.collect()  # Free memory

            # Initialize the data structures for MBM
            print("Initializing training data structures for MBM...")
            rfcs_whole_Data = Data(rfcs_array_train)
            obs_whole_Data = Data(obs_array_train)

                
            

            print("NaN check - rfcs_whole_Data.data:", np.isnan(rfcs_whole_Data.data).any())
            print("NaN check - obs_whole_Data.data:", np.isnan(obs_whole_Data.data).any())

            # Train the MBM postprocessor
            print("Training MBM postprocessor...")
            essacc = MBM.EnsembleAbsCRPSTruncCorrection()
            print("Na training MBM, alles nog goed.")
            essacc.train(obs_whole_Data, rfcs_whole_Data, ntrial=1)
            print("MBM training complete.")

            # Clear memory
            del rfcs_array_train, obs_array_train
            gc.collect()

            # Load testing forecast data for the current lead time
            rfcs_array_test = load_forecast_data(rfcs_test_file, base_dir, params, lead_time)
            print("NaN check - rfcs_array_test:", np.isnan(rfcs_array_test).any())
            gc.collect()

            # Load testing observation data for the current lead time
            obs_array_test = load_observation_data(obs_test_file, base_dir, lead_time)
            print("NaN check - obs_array_test:", np.isnan(obs_array_test).any())
            gc.collect()
            


            # Apply MBM postprocessor to the test data
            print("Applying MBM postprocessor to testing data...")
            test_whole = Data(rfcs_array_test)
            testMBM_whole = essacc(test_whole)[:,:,:,:,:,:]
            print(f"MBM testing complete for lead time {lead_time}.")
            
            # Apply MBM postprocessor to the test data
            print("Applying MBM postprocessor to testing data...")
            test_whole = Data(rfcs_array_test)
            testMBM_whole = essacc(test_whole)[:,:,:,:,:,:]
            print(f"MBM testing complete for lead time {lead_time}.")

            # ➕ Voeg dit toe om de relevante SSRD6 arrays eruit te halen:
            ssrd6_index = params.index('ssrd6')
            
            print("rfcs_array_test shape:", rfcs_array_test.shape)
            print("ssrd6_index:", ssrd6_index)
            print("params:", params)

            forecast_ssrd6 = rfcs_array_test[ssrd6_index, :, 0, 0, 0, :, :]  # shape: (time, lat, lon)
            print("testMBM_whole shape:", testMBM_whole.shape)

            # Alleen als testMBM_whole exact dezelfde structuur heeft als rfcs_array_test
            if testMBM_whole.shape[0] == len(params):
                mbm_prediction_ssrd6 = testMBM_whole[ssrd6_index, :, 0, 0, 0, :, :]
            else:
                # testMBM_whole bevat alleen ssrd6
                mbm_prediction_ssrd6 = testMBM_whole[0, :, 0, 0, 0, :, :]

            observation_ssrd6 = obs_array_test[0, :, 0, 0, 0, :, :]  # shape: (time, lat, lon)
            #print('voor np')
            #print("rfcs_array_test shape:", rfcs_array_test.shape)
            #print("ssrd6_index:", ssrd6_index)
            #print("params:", params)
            #print('begin met printen')
            # ➕ En sla ze op voor plotting in notebook:
            np.save(os.path.join(output_dir, f'forecast_ssrd6_lt{lead_time}.npy'), forecast_ssrd6)
            np.save(os.path.join(output_dir, f'mbm_prediction_ssrd6_lt{lead_time}.npy'), mbm_prediction_ssrd6)
            np.save(os.path.join(output_dir, f'observation_ssrd6_lt{lead_time}.npy'), observation_ssrd6)


            # ➡️ Dan pas het opslaan van de volledige MBM-output:
            output_path = os.path.join(output_dir, f'MBM_Trunc_WS_{lead_time}.npy')
            np.save(output_path, testMBM_whole)

            
            # End timing for this lead time
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Save the result for the current lead time
            output_path = os.path.join(output_dir, f'MBM_Trunc_WS_{lead_time}.npy')
            np.save(output_path, testMBM_whole)
            print(f"Results saved to {output_path}")

            # Write the elapsed time to the log file
            time_log_file.write(f"Lead time {lead_time}: {elapsed_time:.2f} seconds\n")
            time_log_file.flush()  # Ensure the log is written immediately
            print(f"Lead time {lead_time} took {elapsed_time:.2f} seconds")

        except Exception as e:
            print(f"An error occurred for lead time {lead_time}: {e}")
            time_log_file.write(f"Lead time {lead_time}: ERROR - {str(e)}\n")
            time_log_file.flush()


print("All lead times processed.")