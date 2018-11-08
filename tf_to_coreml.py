import tfcoreml as tf_converter
tf_converter.convert(tf_model_path = '2017_09_04_22_37_train_kin_no_drift_posenet.pb',
                     mlmodel_path = '2017_09_04_22_37_train_kin_no_drift_posenet.mlmodel',
                     output_feature_names = ['cls3_fc_pose_xyz/BiasAdd', 'cls3_fc_pose_wpqr/BiasAdd'],
                     input_name_shape_dict = {'input:0' : [1, 224, 224, 3]})
