SPACETRACK_USERNAME=''
SPACETRACK_PASSWORD=''

import orbit_prediction.spacetrack_etl as etl
import pandas as pd

spacetrack_client = etl.build_space_track_client(SPACETRACK_USERNAME,
                                                 SPACETRACK_PASSWORD)

spacetrack_etl = etl.SpaceTrackETL(spacetrack_client)

iss_orbit_data = spacetrack_etl.build_leo_df(norad_ids=['25544'],
                                             last_n_days=30,
                                             only_latest=None)

iss_orbit_data.to_csv(r'./output/my_data.csv')

# iss_orbit_data = pd.read_csv(r'D:\workdir\sandbox\spacetech-ssa-win\orbit_prediction\output\my_data.csv')
import orbit_prediction.build_training_data as training

physics_model_predicted_orbits = training.predict_orbits(iss_orbit_data,
                                                         last_n_days=None,
                                                         n_pred_days=3)

physics_model_errors = training.calc_physics_error(physics_model_predicted_orbits)


import orbit_prediction.ml_model as ml

train_test_data = ml.build_train_test_sets(physics_model_errors, test_size=0.2)

gbrt_params = {'tree_method': 'hist'}
# gbrt_params = {
#     'booster': 'gbtree',
#     'tree_method': 'gpu_hist',
#     'max_depth': 10,
#     'min_child_weight': 3
# }
physics_error_model = ml.train_models(train_test_data, params=gbrt_params)

res = physics_error_model.eval_models(train_test_data['X_test'],
                                train_test_data['y_test'])
print(res)
res.to_csv(r'./output/my_data_result.csv')
