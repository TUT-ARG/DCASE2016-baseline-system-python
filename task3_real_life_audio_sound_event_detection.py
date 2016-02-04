#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCASE 2016::Real-life Audio Sound Event Detection / Baseline System
# Copyright (C) 2015-2016 Toni Heittola (toni.heittola@tut.fi) / TUT
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

from src.ui import *
from src.general import *
from src.features import *
from src.sound_event_detection import *
from src.dataset import *
from src.evaluation import *
from src.files import *

import yaml
import numpy
import csv
import warnings
import argparse
import textwrap
import math

from sklearn import mixture

from IPython import embed

__version_info__ = ('0', '7', '1')
__version__ = '.'.join(__version_info__)


def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2016
            Task 3: Real-life Audio Sound Event Detection
            Baseline System
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Toni Heittola ( toni.heittola@tut.fi )

            System description
                This is an baseline implementation for the D-CASE 2016, task 3 - Real-life audio sound event detection.
                The system has binary classifier for each included sound event class. The GMM classifier is trained with
                the positive and negative examples from the mixture signals, and classification is done between these
                two models as likelihood ratio. Acoustic features are MFCC+Delta+Acceleration (MFCC0 omitted).

        '''))

    parser.add_argument("-development", help="Use the system in the development mode", action='store_true',
                        default=False, dest='development')
    parser.add_argument("-challenge", help="Use the system in the challenge mode", action='store_true',
                        default=False, dest='challenge')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    args = parser.parse_args()

    # Load parameters from config file
    params = load_parameters('task3_real_life_audio_sound_event_detection.yaml')
    params = process_parameters(params)

    title("DCASE 2016::Real-life Audio Sound Event Detection / Baseline System")

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
        section_header('Feature extraction [Development data]')

        # Collect files in train sets
        files = []
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            for item_id, item in enumerate(dataset.train(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
            for item_id, item in enumerate(dataset.test(fold)):
                if item['file'] not in files:
                    files.append(item['file'])

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
        section_header('Feature normalizer [Development data]')

        do_feature_normalization(dataset=dataset,
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 feature_normalizer_path=params['path']['feature_normalizers'],
                                 feature_path=params['path']['features'],
                                 overwrite=params['general']['overwrite'])

        foot()

    # System training
    # ==================================================
    if params['flow']['train_system']:
        section_header('System training    [Development data]')

        do_system_training(dataset=dataset,
                           dataset_evaluation_mode=dataset_evaluation_mode,
                           model_path=params['path']['models'],
                           feature_normalizer_path=params['path']['feature_normalizers'],
                           feature_path=params['path']['features'],
                           hop_length_seconds=params['features']['hop_length_seconds'],
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
            section_header('System testing     [Development data]')

            do_system_testing(dataset=dataset,
                              dataset_evaluation_mode=dataset_evaluation_mode,
                              result_path=params['path']['results'],
                              model_path=params['path']['models'],
                              feature_params=params['features'],
                              detector_params=params['detector'],
                              classifier_method=params['classifier']['method'],
                              overwrite=params['general']['overwrite']
                              )
            foot()

        # System evaluation
        # ==================================================
        if params['flow']['evaluate_system']:
            section_header('System evaluation  [Development data]')

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
            section_header('System testing     [Challenge data]')

            do_system_testing(dataset=challenge_dataset,
                              dataset_evaluation_mode=dataset_evaluation_mode,
                              result_path=params['path']['challenge_results'],
                              model_path=params['path']['models'],
                              feature_params=params['features'],
                              detector_params=params['detector'],
                              overwrite=True
                              )
            foot()

            print " "
            print "Your results for the challenge data are stored at ["+params['path']['challenge_results']+"]"
            print " "

def process_parameters(params):
    params['features']['mfcc']['win_length'] = int(params['features']['win_length_seconds'] * params['features']['fs'])
    params['features']['mfcc']['hop_length'] = int(params['features']['hop_length_seconds'] * params['features']['fs'])

    # Copy parameters for current classifier method
    params['classifier']['parameters'] = params['classifier_parameters'][params['classifier']['method']]

    params['features']['hash'] = get_parameter_hash(params['features'])
    params['classifier']['hash'] = get_parameter_hash(params['classifier'])
    params['detector']['hash'] = get_parameter_hash(params['detector'])

    params['path']['features'] = os.path.join(params['path']['base'], params['path']['features'], params['features']['hash'])
    params['path']['feature_normalizers'] = os.path.join(params['path']['base'], params['path']['feature_normalizers'], params['features']['hash'])
    params['path']['models'] = os.path.join(params['path']['base'], params['path']['models'], params['features']['hash'], params['classifier']['hash'])
    params['path']['results'] = os.path.join(params['path']['base'], params['path']['results'], params['features']['hash'], params['classifier']['hash'], params['detector']['hash'])
    return params


def get_feature_filename(audio_file, path, extension='cpickle'):
    return os.path.join(path, 'sequence_' + os.path.splitext(audio_file)[0] + '.' + extension)


def get_feature_normalizer_filename(fold, scene_label, path, extension='cpickle'):
    return os.path.join(path, 'scale_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)


def get_model_filename(fold, scene_label, path, extension='cpickle'):
    return os.path.join(path, 'model_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)


def get_result_filename(fold, scene_label, path, extension='txt'):
    if fold == 0:
        return os.path.join(path, 'results_' + str(scene_label) + '.' + extension)
    else:
        return os.path.join(path, 'results_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)


def do_feature_extraction(files, dataset, feature_path, params, overwrite=False):
    # Check that target path exists, create if not
    check_path(feature_path)

    for file_id, audio_filename in enumerate(files):
        # Get feature filename
        current_feature_file = get_feature_filename(audio_file=os.path.split(audio_filename)[1], path=feature_path)

        progress(title='Extracting [sequences]',
                 percentage=(float(file_id) / len(files)),
                 note=os.path.split(audio_filename)[1])

        if not os.path.isfile(current_feature_file) or overwrite:
            # Load audio
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
        for scene_id, scene_label in enumerate(dataset.scene_labels):
            current_normalizer_file = get_feature_normalizer_filename(fold=fold, scene_label=scene_label, path=feature_normalizer_path)
            
            if not os.path.isfile(current_normalizer_file) or overwrite:
                # Collect sequence files from scene class
                files = []                
                for item_id, item in enumerate(dataset.train(fold, scene_label=scene_label)):
                    if item['file'] not in files:
                        files.append(item['file'])

                file_count = len(files)

                # Initialize statistics
                normalizer = FeatureNormalizer()

                for file_id, audio_filename in enumerate(files):
                    progress(title='Collecting data',
                             fold=fold,
                             percentage=(float(file_id) / file_count),
                             note=os.path.split(audio_filename)[1])

                    # Load features
                    feature_filename = get_feature_filename(audio_file=os.path.split(audio_filename)[1], path=feature_path)
                    if os.path.isfile(feature_filename):
                        feature_data = load_data(feature_filename)['stat']
                    else:
                        raise IOError("Features missing [%s]" % audio_filename)

                    # Accumulate statistics
                    normalizer.accumulate(feature_data)

                # Calculate normalization factors
                normalizer.finalize()

                # Save
                save_data(current_normalizer_file, normalizer)


def do_system_training(dataset, dataset_evaluation_mode, model_path, feature_normalizer_path, feature_path,
                       hop_length_seconds, classifier_params, classifier_method='gmm', overwrite=False):
    if classifier_method != 'gmm':
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(model_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        for scene_id, scene_label in enumerate(dataset.scene_labels):
            current_model_file = get_model_filename(fold=fold, scene_label=scene_label, path=model_path)
            if not os.path.isfile(current_model_file) or overwrite:

                # Load normalizer
                feature_normalizer_filename = get_feature_normalizer_filename(fold=fold, scene_label=scene_label, path=feature_normalizer_path)
                if os.path.isfile(feature_normalizer_filename):
                    normalizer = load_data(feature_normalizer_filename)
                else:
                    raise IOError("Feature normalizer missing [%s]" % feature_normalizer_filename)

                # Initialize model container
                model_container = {'normalizer': normalizer, 'models': {}}

                # Restructure training data in to structure[files][events]
                ann = {}
                for item_id, item in enumerate(dataset.train(fold=fold, scene_label=scene_label)):
                    filename = os.path.split(item['file'])[1]
                    if filename not in ann:
                        ann[filename] = {}
                    if item['event_label'] not in ann[filename]:
                        ann[filename][item['event_label']] = []
                    ann[filename][item['event_label']].append((item['event_onset'], item['event_offset']))

                # Collect training examples
                data_positive = {}
                data_negative = {}
                file_count = len(ann)
                for item_id, audio_filename in enumerate(ann):
                    progress(title='Collecting data',
                             fold=fold,
                             percentage=(float(item_id) / file_count),
                             note=scene_label+" / "+os.path.split(audio_filename)[1])

                    # Load features
                    feature_filename = get_feature_filename(audio_file=audio_filename, path=feature_path)
                    if os.path.isfile(feature_filename):
                        feature_data = load_data(feature_filename)['feat']
                    else:
                        raise IOError("Features missing [%s]" % feature_filename)

                    # Normalize features
                    feature_data = model_container['normalizer'].normalize(feature_data)

                    for event_label in ann[audio_filename]:
                        positive_mask = numpy.zeros((feature_data.shape[0]), dtype=bool)

                        for event in ann[audio_filename][event_label]:
                            start_frame = math.floor(event[0] / hop_length_seconds)
                            stop_frame = math.ceil(event[1] / hop_length_seconds)

                            if stop_frame > feature_data.shape[0]:
                                stop_frame = feature_data.shape[0]

                            positive_mask[start_frame:stop_frame] = True

                        # Store positive examples
                        if event_label not in data_positive:
                            data_positive[event_label] = feature_data[positive_mask, :]
                        else:
                            data_positive[event_label] = numpy.vstack((data_positive[event_label], feature_data[positive_mask, :]))

                        # Store negative examples
                        if event_label not in data_negative:
                            data_negative[event_label] = feature_data[~positive_mask, :]
                        else:
                            data_negative[event_label] = numpy.vstack((data_negative[event_label], feature_data[~positive_mask, :]))

                # Train models for each class
                for event_label in data_positive:
                    progress(title='Train models',
                             fold=fold,
                             note=scene_label+" / "+event_label)
                    if classifier_method == 'gmm':
                        model_container['models'][event_label] = {}
                        model_container['models'][event_label]['positive'] = mixture.GMM(**classifier_params).fit(data_positive[event_label])
                        model_container['models'][event_label]['negative'] = mixture.GMM(**classifier_params).fit(data_negative[event_label])
                    else:
                        raise ValueError("Unknown classifier method ["+classifier_method+"]")

                # Save models
                save_data(current_model_file, model_container)


def do_system_testing(dataset, dataset_evaluation_mode, result_path, model_path, feature_params, detector_params, classifier_method='gmm',
                      overwrite=False):

    if classifier_method != 'gmm':
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(result_path)
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        for scene_id, scene_label in enumerate(dataset.scene_labels):
            current_result_file = get_result_filename(fold=fold, scene_label=scene_label, path=result_path)
            if not os.path.isfile(current_result_file) or overwrite:
                results = []

                # Load class model container
                model_filename = get_model_filename(fold=fold, scene_label=scene_label, path=model_path)
                if os.path.isfile(model_filename):
                    model_container = load_data(model_filename)
                else:
                    raise IOError("Model file not found [%s]" % model_filename)


                file_count = len(dataset.test(fold, scene_label=scene_label))
                for file_id, item in enumerate(dataset.test(fold=fold, scene_label=scene_label)):
                    progress(title='Testing',
                             fold=fold,
                             percentage=(float(file_id) / file_count),
                             note=scene_label+" / "+os.path.split(item['file'])[1])

                    # Load audio
                    if os.path.isfile(dataset.relative_to_absolute_path(item['file'])):
                        y, fs = load_audio(filename=item['file'], mono=True, fs=feature_params['fs'])
                    else:
                        raise IOError("Audio file not found [%s]" % item['file'])

                    # Extract features
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

                    current_results = event_detection(feature_data=feature_data,
                                                      model_container=model_container,
                                                      hop_length_seconds=feature_params['hop_length_seconds'],
                                                      smoothing_window_length_seconds=detector_params['smoothing_window_length'],
                                                      decision_threshold=detector_params['decision_threshold'],
                                                      minimum_event_length=detector_params['minimum_event_length'],
                                                      minimum_event_gap=detector_params['minimum_event_gap'])

                    # Store the result
                    for event in current_results:
                        results.append((dataset.absolute_to_relative(item['file']), event[0], event[1], event[2] ))

                # Save testing results
                with open(current_result_file, 'wt') as f:
                    writer = csv.writer(f, delimiter='\t')
                    for result_item in results:
                        writer.writerow(result_item)


def do_system_evaluation(dataset, dataset_evaluation_mode, result_path):
    
    # Set warnings off, sklearn metrics will trigger warning for classes without
    # predicted samples in F1-scoring. This is just to keep printing clean.
    warnings.simplefilter("ignore")
    overall_metrics_per_scene = {}

    for scene_id, scene_label in enumerate(dataset.scene_labels):
        if scene_label not in overall_metrics_per_scene:
            overall_metrics_per_scene[scene_label] = {}

        dcase2016_segment_based_metric = DCASE2016_EventDetection_SegmentBasedMetrics(class_list=dataset.event_labels(scene_label=scene_label))
        dcase2016_event_based_metric = DCASE2016_EventDetection_EventBasedMetrics(class_list=dataset.event_labels(scene_label=scene_label))

        for fold in dataset.folds(mode=dataset_evaluation_mode):
            #dcase2016_segment_based_metric_fold = DCASE2016_EventDetection_SegmentBasedMetrics(class_list=dataset.event_labels(scene_label=scene_label))
            results = []
            result_filename = get_result_filename(fold=fold, scene_label=scene_label, path=result_path)

            if os.path.isfile(result_filename):
                with open(result_filename, 'rt') as f:
                    for row in csv.reader(f, delimiter='\t'):
                        results.append(row)
            else:
                raise IOError("Result file not found [%s]" % result_filename)

            for file_id, item in enumerate(dataset.test(fold,scene_label=scene_label)):
                current_file_results = []
                for result_line in results:
                    if result_line[0] == dataset.absolute_to_relative(item['file']):
                        current_file_results.append(
                            {'file': result_line[0],
                             'event_onset': float(result_line[1]),
                             'event_offset': float(result_line[2]),
                             'event_label': result_line[3]
                             }
                        )
                meta = dataset.file_meta(dataset.absolute_to_relative(item['file']))

                dcase2016_segment_based_metric.evaluate(current_file_results, meta)
                dcase2016_event_based_metric.evaluate(current_file_results, meta)
                #dcase2016_segment_based_metric_fold.evaluate(current_file_results, meta)
            #res = dcase2016_segment_based_metric_fold.results()
            #print scene_label, "fold="+str(fold), "er="+str(res['overall']['ER']), "F="+str(res['overall']['F']*100)
        overall_metrics_per_scene[scene_label]['segment_based_metrics'] = dcase2016_segment_based_metric.results()
        overall_metrics_per_scene[scene_label]['event_based_metrics'] = dcase2016_event_based_metric.results()


    print "  Evaluation over %d folds" % dataset.fold_count
    print " "
    print "  Results per scene "
    print "  {:18s} | {:4s} |  | {:39s}  ".format('', 'Main', 'Secondary metrics')
    print "  {:18s} | {:4s} |  | {:34s} | {:13s} | {:13s} |  {:13s} ".format('', '', 'Seg/Overall','Seg/Class', 'Event/Overall','Event/Class')
    print "  {:18s} | {:4s} |  | {:6s} : {:4s} : {:4s} : {:4s} : {:4s} | {:6s} : {:4s} | {:6s} : {:4s} | {:6s} : {:4s} |".format('Scene', 'ER', 'F1', 'ER', 'ER/S', 'ER/D', 'ER/I', 'F1', 'ER', 'F1', 'ER', 'F1', 'ER')
    print "  -------------------+------+  +--------+------+------+------+------+--------+------+--------+------+--------+------+"
    averages = {
        'segment_based_metrics': {
            'overall': {
                'ER': [],
                'F': [],
            },
            'class_wise_average': {
                'ER': [],
                'F': [],
            }
        },
        'event_based_metrics': {
            'overall': {
                'ER': [],
                'F': [],
            },
            'class_wise_average': {
                'ER': [],
                'F': [],
            }
        },
    }
    for scene_id, scene_label in enumerate(dataset.scene_labels):
        print "  {:18s} | {:3.2f} |  | {:4.1f} % : {:3.2f} : {:3.2f} : {:3.2f} : {:3.2f} | {:4.1f} % : {:3.2f} | {:4.1f} % : {:3.2f} | {:4.1f} % : {:3.2f} |".format(scene_label,
                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['overall']['ER'],
                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['overall']['F'] * 100,
                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['overall']['ER'],
                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['overall']['S'],
                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['overall']['D'],
                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['overall']['I'],
                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise_average']['F']*100,
                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise_average']['ER'],
                                        overall_metrics_per_scene[scene_label]['event_based_metrics']['overall']['F']*100,
                                        overall_metrics_per_scene[scene_label]['event_based_metrics']['overall']['ER'],
                                        overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise_average']['F']*100,
                                        overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise_average']['ER'],
                                                                     )
        averages['segment_based_metrics']['overall']['ER'].append(overall_metrics_per_scene[scene_label]['segment_based_metrics']['overall']['ER'])
        averages['segment_based_metrics']['overall']['F'].append(overall_metrics_per_scene[scene_label]['segment_based_metrics']['overall']['F'])
        averages['segment_based_metrics']['class_wise_average']['ER'].append(overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise_average']['ER'])
        averages['segment_based_metrics']['class_wise_average']['F'].append(overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise_average']['F'])
        averages['event_based_metrics']['overall']['ER'].append(overall_metrics_per_scene[scene_label]['event_based_metrics']['overall']['ER'])
        averages['event_based_metrics']['overall']['F'].append(overall_metrics_per_scene[scene_label]['event_based_metrics']['overall']['F'])
        averages['event_based_metrics']['class_wise_average']['ER'].append(overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise_average']['ER'])
        averages['event_based_metrics']['class_wise_average']['F'].append(overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise_average']['F'])

    print "  -------------------+------+  +--------+------+------+------+------+--------+------+--------+------+--------+------+"
    print "  {:18s} | {:3.2f} |  | {:4.1f} % : {:3.2f} : {:18s} | {:4.1f} % : {:3.2f} | {:4.1f} % : {:3.2f} | {:4.1f} % : {:3.2f} |".format('Average',
                                        numpy.mean(averages['segment_based_metrics']['overall']['ER']),
                                        numpy.mean(averages['segment_based_metrics']['overall']['F'])*100,
                                        numpy.mean(averages['segment_based_metrics']['overall']['ER']),
                                        ' ',
                                        numpy.mean(averages['segment_based_metrics']['class_wise_average']['F'])*100,
                                        numpy.mean(averages['segment_based_metrics']['class_wise_average']['ER']),
                                        numpy.mean(averages['event_based_metrics']['overall']['F'])*100,
                                        numpy.mean(averages['event_based_metrics']['overall']['ER']),
                                        numpy.mean(averages['event_based_metrics']['class_wise_average']['F'])*100,
                                        numpy.mean(averages['event_based_metrics']['class_wise_average']['ER']),
                                                    )

    print "  "
    # Restore warnings to default settings
    warnings.simplefilter("default")
    print "  Results per events "

    for scene_id, scene_label in enumerate(dataset.scene_labels):
        print "  "
        print "  "+scene_label.upper()
        print "  {:20s} | {:27s} |  | {:13s} ".format('','Segment-based', 'Event-based')
        print "  {:20s} | {:4s} : {:4s} : {:6s} : {:4s} |  | {:4s} : {:4s} : {:6s} : {:4s} |".format('Event', 'Nref', 'Nsys', 'F1', 'ER', 'Nref', 'Nsys', 'F1', 'ER')
        print "  ---------------------+------+------+--------+------+  +------+------+--------+------+"
        seg_Nref = 0
        seg_Nsys = 0

        event_Nref = 0
        event_Nsys = 0
        for event_label in sorted(overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise']):
            print "  {:20s} | {:4d} : {:4d} : {:4.1f} % : {:3.2f} |  | {:4d} : {:4d} : {:4.1f} % : {:3.2f} |".format(event_label,
                                                                        int(overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise'][event_label]['Nref']),
                                                                        int(overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise'][event_label]['Nsys']),
                                                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise'][event_label]['F']*100,
                                                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise'][event_label]['ER'],
                                                                        int(overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise'][event_label]['Nref']),
                                                                        int(overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise'][event_label]['Nsys']),
                                                                        overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise'][event_label]['F']*100,
                                                                        overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise'][event_label]['ER'])
            seg_Nref += int(overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise'][event_label]['Nref'])
            seg_Nsys += int(overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise'][event_label]['Nsys'])

            event_Nref += int(overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise'][event_label]['Nref'])
            event_Nsys += int(overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise'][event_label]['Nsys'])
        print "  ---------------------+------+------+--------+------+  +------+------+--------+------+"
        print "  {:20s} | {:4d} : {:4d} : {:13s} |  | {:4d} : {:4d} : {:13s} |".format('Sum',
                                                                        seg_Nref,
                                                                        seg_Nsys,
                                                                        '',
                                                                        event_Nref,
                                                                        event_Nsys,
                                                                        '')
        print "  {:20s} | {:4s}   {:4s} : {:4.1f} % : {:3.2f} |  | {:4s}   {:4s} : {:4.1f} % : {:3.2f} |".format('Average',
                                                                        '', '',
                                                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise_average']['F']*100,
                                                                        overall_metrics_per_scene[scene_label]['segment_based_metrics']['class_wise_average']['ER'],
                                                                        '', '',
                                                                        overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise_average']['F']*100,
                                                                        overall_metrics_per_scene[scene_label]['event_based_metrics']['class_wise_average']['ER'])
        print "  "

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
