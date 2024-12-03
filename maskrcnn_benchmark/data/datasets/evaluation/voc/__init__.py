import logging

from .voc_eval import do_voc_evaluation,do_voc_evaluate
from .evaluate_voc import do_voc_evaluation1

def voc_evaluation(dataset, predictions, output_folder,voc_evaluate, box_only, **_):
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    if box_only:
        logger.warning("voc evaluation doesn't support box_only, ignored.")
    logger.info("performing voc evaluation, ignored iou_types.")
    if voc_evaluate:
        #do_voc_evaluate(dataset, predictions)
        return do_voc_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
        )
    else:
        return do_voc_evaluation1(
            dataset=dataset,
            predictions=predictions,
        )
