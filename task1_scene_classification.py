#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCASE 2016::Acoustic Scene Classification / Baseline System
# Copyright (C) 2015 Toni Heittola (toni.heittola@tut.fi) / TUT
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from src.general import *
from src.files import *
from src.features import *
from src.dataset import *
from src.evaluation import *
from src.ui import *

import yaml
import numpy
import csv
import warnings
import argparse
import textwrap

from sklearn import mixture, metrics

from IPython import embed

__version_info__ = ('0', '7', '0')
__version__ = '.'.join(__version_info__)



def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2016
            Task 1: Acoustic Scene Classification
            Baseline system
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Toni Heittola ( toni.heittola@tut.fi )

            System description
                This is an baseline implementation for D-CASE 2016 challenge acoustic scene classification task.
                Features: MFCC (static+delta+acceleration)
                Classifier: GMM

        '''))

    # Setup argument handling
    parser.add_argument("-development", help="Use the system in the development mode", action='store_true',
                        default=False, dest='development')
    parser.add_argument("-challenge", help="Use the system in the challenge mode", action='store_true',
                        default=False, dest='challenge')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    args = parser.parse_args()

    # Load parameters from config file
    params = load_parameters('task1_scene_classification.yaml')
    params = process_parameters(params)

    title("DCASE 2016::Acoustic Scene Classification / Baseline System")

    # Check if mode is defined
    if not (args.development or args.challenge):
        args.development = True
        args.challenge = False

    dataset_evaluation_mode = 'folds'
    if args.development and not args.challenge:
        print "Running system in development mode"
        dataset_evaluation_mode = 'folds'
    elif not args.development and args.challenge:
        print "Running system in challenge mode"
        dataset_evaluation_mode = 'full'

    # Get dataset container class
    dataset = eval(params['general']['development_dataset'])(data_path=params['path']['data'])

    # Fetch data over internet and setup the data
    # ==================================================
    if params['flow']['initialize']:
        dataset.fetch()

    # Extract features for all audio files in the dataset
    # ==================================================
    if params['flow']['extract_features']:
        section_header('Feature extraction')

        # Collect files in train sets
        files = []
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            for item_id, item in enumerate(dataset.train(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
            for item_id, item in enumerate(dataset.test(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
        files = sorted(files)

        # Go through files and make sure all features are extracted
        do_feature_extraction(files=files,
                              dataset=dataset,
                              feature_path=params['path']['features'],
                              params=params['features'],
                              overwrite=params['general']['overwrite'])

        foot()

    # Prepare feature normalizers
    # ==================================================
    if params['flow']['feature_normalizer']:
        section_header('Feature normalizer')

        do_feature_normalization(dataset=dataset,
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 feature_normalizer_path=params['path']['feature_normalizers'],
                                 feature_path=params['path']['features'],
                                 overwrite=params['general']['overwrite'])

        foot()

    # System training
    # ==================================================
    if params['flow']['train_system']:
        section_header('System training')

        do_system_training(dataset=dataset,
                           dataset_evaluation_mode=dataset_evaluation_mode,
                           model_path=params['path']['models'],
                           feature_normalizer_path=params['path']['feature_normalizers'],
                           feature_path=params['path']['features'],
                           classifier_params=params['classifier']['parameters'],
                           classifier_method=params['classifier']['method'],
                           overwrite=params['general']['overwrite']
                           )

        foot()

    # System evaluation in development mode
    if args.development and not args.challenge:

        # System testing
        # ==================================================
        if params['flow']['test_system']:
            section_header('System testing')

            do_system_testing(dataset=dataset,
                              dataset_evaluation_mode=dataset_evaluation_mode,
                              feature_path=params['path']['features'],
                              result_path=params['path']['results'],
                              model_path=params['path']['models'],
                              feature_params=params['features'],
                              #evaluation_params=params['evaluation'],
                              classifier_method=params['classifier']['method'],
                              overwrite=params['general']['overwrite']
                              )
            
            foot()

        # System evaluation
        # ==================================================
        if params['flow']['evaluate_system']:
            section_header('System evaluation')

            do_system_evaluation(dataset=dataset,
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 result_path=params['path']['results'])

            foot()

    # System evaluation with challenge data
    elif not args.development and args.challenge:
        # Fetch data over internet and setup the data
        challenge_dataset = eval(params['general']['challenge_dataset'])()

        if params['flow']['initialize']:
            challenge_dataset.fetch()

        # System testing
        if params['flow']['test_system']:
            section_header('System testing with challenge data')

            do_system_testing(dataset=challenge_dataset,
                              dataset_evaluation_mode=dataset_evaluation_mode,
                              feature_path=params['path']['features'],
                              result_path=params['path']['challenge_results'],
                              model_path=params['path']['models'],
                              feature_params=params['features'],
                              #evaluation_params=params['evaluation'],
                              classifier_method=params['classifier']['method'],
                              overwrite=True
                              )

            foot()

            print " "
            print "Your results for the challenge data are stored at ["+params['path']['challenge_results']+"]"
            print " "
    return 0


def process_parameters(params):
    params['features']['mfcc']['win_length'] = int(params['features']['win_length_seconds'] * params['features']['fs'])
    params['features']['mfcc']['hop_length'] = int(params['features']['hop_length_seconds'] * params['features']['fs'])

    # Copy parameters for current classifier method
    params['classifier']['parameters'] = params['classifier_parameters'][params['classifier']['method']]

    params['features']['hash'] = get_parameter_hash(params['features'])
    params['classifier']['hash'] = get_parameter_hash(params['classifier'])
    #params['evaluation']['hash'] = get_parameter_hash(params['evaluation'])
    
    params['path']['features'] = os.path.join(params['path']['base'], params['path']['features'],
                                              params['features']['hash'])
    params['path']['feature_normalizers'] = os.path.join(params['path']['base'], params['path']['feature_normalizers'],
                                                         params['features']['hash'])
    params['path']['models'] = os.path.join(params['path']['base'], params['path']['models'],
                                            params['features']['hash'], params['classifier']['hash'])
    params['path']['results'] = os.path.join(params['path']['base'], params['path']['results'],
                                             params['features']['hash'], params['classifier']['hash'])#,
                                             #params['evaluation']['hash'])
    return params


def get_feature_filename(audio_file, path, extension='cpickle'):
    audio_filename = os.path.split(audio_file)[1]
    return os.path.join(path, os.path.splitext(audio_filename)[0] + '.' + extension)


def get_feature_normalizer_filename(fold, path, extension='cpickle'):
    return os.path.join(path, 'scale_fold' + str(fold) + '.' + extension)


def get_model_filename(fold, path, extension='cpickle'):
    return os.path.join(path, 'model_fold' + str(fold) + '.' + extension)


def get_result_filename(fold, path, extension='txt'):
    return os.path.join(path, 'results_fold' + str(fold) + '.' + extension)


def do_feature_extraction(files, dataset, feature_path, params, overwrite=False):
        # Check that target path exists, create if not
        check_path(feature_path)

        for file_id, audio_filename in enumerate(files):
            # Get feature filename
            current_feature_file = get_feature_filename(audio_file=os.path.split(audio_filename)[1], path=feature_path)

            progress(title='Extracting',
                     percentage=(float(file_id) / len(files)),
                     note=os.path.split(audio_filename)[1])

            if not os.path.isfile(current_feature_file) or overwrite:
                # Load audio data
                if os.path.isfile(dataset.relative_to_absolute_path(audio_filename)):
                    y, fs = load_audio(filename=dataset.relative_to_absolute_path(audio_filename), mono=True, fs=params['fs'])
                else:
                    raise IOError("Audio file not found [%s]" % audio_filename)

                # Extract features
                feature_data = feature_extraction(y=y,
                                                  fs=fs,
                                                  include_mfcc0=params['include_mfcc0'],
                                                  include_delta=params['include_delta'],
                                                  include_acceleration=params['include_acceleration'],
                                                  mfcc_params=params['mfcc'],
                                                  delta_params=params['mfcc_delta'],
                                                  acceleration_params=params['mfcc_acceleration'])
                # Save
                save_data(current_feature_file, feature_data)


def do_feature_normalization(dataset, dataset_evaluation_mode, feature_normalizer_path, feature_path, overwrite=False):
    # Check that target path exists, create if not
    check_path(feature_normalizer_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_normalizer_file = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)

        if not os.path.isfile(current_normalizer_file) or overwrite:
            # Initialize statistics
            file_count = len(dataset.train(fold))
            normalizer = FeatureNormalizer()

            for item_id, item in enumerate(dataset.train(fold)):
                progress(title='Collecting data',
                         fold=fold,
                         percentage=(float(item_id) / file_count),
                         note=os.path.split(item['file'])[1])
                # Load features
                if os.path.isfile(get_feature_filename(audio_file=item['file'], path=feature_path)):
                    feature_data = load_data(get_feature_filename(audio_file=item['file'], path=feature_path))['stat']
                else:
                    raise IOError("Features missing [%s]" % (item['file']))

                # Accumulate statistics
                normalizer.accumulate(feature_data)
            
            # Calculate normalization factors
            normalizer.finalize()

            # Save
            save_data(current_normalizer_file, normalizer)


def do_system_training(dataset, dataset_evaluation_mode, model_path, feature_normalizer_path, feature_path,
                       classifier_params, classifier_method='gmm', overwrite=False):

    if classifier_method != 'gmm':
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(model_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_model_file = get_model_filename(fold=fold, path=model_path)
        if not os.path.isfile(current_model_file) or overwrite:
            # Load normalizer
            feature_normalizer_filename = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)
            if os.path.isfile(feature_normalizer_filename):
                normalizer = load_data(feature_normalizer_filename)
            else:
                raise IOError("Feature normalizer missing [%s]" % feature_normalizer_filename)

            # Initialize model container
            model_container = {'normalizer': normalizer, 'models': {}}

            # Collect training examples
            file_count = len(dataset.train(fold))
            data = {}
            for item_id, item in enumerate(dataset.train(fold)):
                progress(title='Collecting data',
                         fold=fold,
                         percentage=(float(item_id) / file_count),
                         note=os.path.split(item['file'])[1])

                # Load features
                feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
                if os.path.isfile(feature_filename):
                    feature_data = load_data(feature_filename)['feat']
                else:
                    raise IOError("Features missing [%s]" % (item['file']))

                # Scale features
                feature_data = model_container['normalizer'].normalize(feature_data)

                # Store features per class label
                if item['scene_label'] not in data:
                    data[item['scene_label']] = feature_data
                else:
                    data[item['scene_label']] = numpy.vstack((data[item['scene_label']], feature_data))

            # Train models for each class
            for label in data:
                progress(title='Train models',
                         fold=fold,
                         note=label)
                if classifier_method == 'gmm':
                    model_container['models'][label] = mixture.GMM(**classifier_params).fit(data[label])
                else:
                    raise ValueError("Unknown classifier method ["+classifier_method+"]")

            # Save models
            save_data(current_model_file, model_container)


def do_system_testing(dataset, dataset_evaluation_mode, feature_path, result_path, model_path, feature_params, classifier_method='gmm', overwrite=False): #evaluation_params, 

    if classifier_method != 'gmm':
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(result_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_result_file = get_result_filename(fold=fold, path=result_path)
        if not os.path.isfile(current_result_file) or overwrite:
            results = []

            # Load class model container
            model_filename = get_model_filename(fold=fold, path=model_path)
            if os.path.isfile(model_filename):
                model_container = load_data(model_filename)
            else:
                raise IOError("Model file not found [%s]" % model_filename)

            file_count = len(dataset.test(fold))
            for file_id, item in enumerate(dataset.test(fold)):
                progress(title='Testing',
                         fold=fold,
                         percentage=(float(file_id) / file_count),
                         note=os.path.split(item['file'])[1])
                
                # Load features
                feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
                
                if os.path.isfile(feature_filename):
                    feature_data = load_data(feature_filename)['feat']
                else:
                  # Load audio
                  if os.path.isfile(dataset.relative_to_absolute_path(item['file'])):
                      y, fs = load_audio(filename=dataset.relative_to_absolute_path(item['file']), mono=True, fs=feature_params['fs'])
                  else:
                      raise IOError("Audio file not found [%s]" % (item['file']))
                  
                  feature_data = feature_extraction(y=y,
                                                    fs=fs,
                                                    include_mfcc0=feature_params['include_mfcc0'],
                                                    include_delta=feature_params['include_delta'],
                                                    include_acceleration=feature_params['include_acceleration'],
                                                    mfcc_params=feature_params['mfcc'],
                                                    delta_params=feature_params['mfcc_delta'],
                                                    acceleration_params=feature_params['mfcc_acceleration'],
                                                    statistics=False)['feat']
                # Normalize features
                feature_data = model_container['normalizer'].normalize(feature_data)

                # Do classification for the block
                if classifier_method == 'gmm':
                    current_result = do_classification_gmm(feature_data, model_container)
                else:
                    raise ValueError("Unknown classifier method ["+classifier_method+"]")

                # Store the result
                results.append((dataset.absolute_to_relative(item['file']), current_result))

            # Save testing results
            with open(current_result_file, 'wt') as f:
                writer = csv.writer(f, delimiter='\t')
                for result_item in results:
                    writer.writerow(result_item)


def do_classification_gmm(feature_data, model_container):
    # Initialize log-likelihood matrix to -inf
    logls = numpy.empty(len(model_container['models']))
    logls.fill(-numpy.inf)

    for label_id, label in enumerate(model_container['models']):
        logls[label_id] = numpy.sum(model_container['models'][label].score(feature_data))

    classification_result_id = numpy.argmax(logls)
    return model_container['models'].keys()[classification_result_id]


def do_system_evaluation(dataset, dataset_evaluation_mode, result_path):
    dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
    results_fold = []
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        dcase2016_scene_metric_fold = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
        results = []
        result_filename = get_result_filename(fold=fold, path=result_path)

        if os.path.isfile(result_filename):
            with open(result_filename, 'rt') as f:
                for row in csv.reader(f, delimiter='\t'):
                    results.append(row)
        else:
            raise IOError("Result file not found [%s]" % result_filename)

        y_true = []
        y_pred = []
        for result in results:
            y_true.append(dataset.file_meta(result[0])[0]['scene_label'])
            y_pred.append(result[1])
        dcase2016_scene_metric.evaluate(system_output=y_pred, annotated_groundtruth=y_true)
        dcase2016_scene_metric_fold.evaluate(system_output=y_pred, annotated_groundtruth=y_true)
        results_fold.append(dcase2016_scene_metric_fold.results())
    results = dcase2016_scene_metric.results()

    print "  File-wise evaluation, over %d folds" % dataset.fold_count
    fold_labels = ''
    separator = '     =====================+======+======+==========+  +'
    if dataset.fold_count > 1:
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            fold_labels += " {:8s} |".format('Fold'+str(fold))
            separator += "==========+"
    print "     {:20s} | {:4s} : {:4s} | {:8s} |  |".format('Scene label', 'Nref', 'Nsys', 'Accuracy')+fold_labels
    print separator
    for label_id, label in enumerate(sorted(results['class_wise_accuracy'])):
        fold_values = ''
        if dataset.fold_count > 1:
            for fold in dataset.folds(mode=dataset_evaluation_mode):
                fold_values += " {:5.1f} %  |".format(results_fold[fold-1]['class_wise_accuracy'][label] * 100)
        print "     {:20s} | {:4d} : {:4d} | {:5.1f} %  |  |".format(label,
                                                                   results['class_wise_data'][label]['Nref'],
                                                                   results['class_wise_data'][label]['Nsys'],
                                                                   results['class_wise_accuracy'][label] * 100)+fold_values
    print separator
    fold_values = ''
    if dataset.fold_count > 1:
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            fold_values += " {:5.1f} %  |".format(results_fold[fold-1]['overall_accuracy'] * 100)

    print "     {:20s} | {:4d} : {:4d} | {:5.1f} %  |  |".format('Overall accuracy',
                                                               results['Nref'],
                                                               results['Nsys'],
                                                               results['overall_accuracy'] * 100)+fold_values

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
