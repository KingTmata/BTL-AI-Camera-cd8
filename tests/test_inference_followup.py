import unittest

from notebooks.kaggle.project8_v05_inference_followup_20261010 import build_inference_jobs, require_full_cache_parity


class InferenceFollowupTests(unittest.TestCase):
    def test_all_selected_models_get_flip_and_three_views_plus_ensembles(self):
        names = ['R12_yolo26s', 'R07_AdamW', 'R04_class_weight_025']
        jobs = build_inference_jobs(names)
        self.assertEqual(len(jobs), 10)
        self.assertEqual(jobs[0], (names[0]+'_single_view_parity', [names[0]], ((1.,False),)))
        by_name = {label:(members,views) for label,members,views in jobs}
        self.assertEqual(len(by_name), 10)
        for name in names:
            self.assertEqual(by_name[name+'_flip'], ([name], ((1.,False),(1.,True))))
            self.assertEqual(by_name[name+'_three_views'], ([name], ((1.,False),(.83,True),(.67,False))))
        self.assertEqual(by_name['ensemble_top2'][0], names[:2])
        self.assertEqual(by_name['ensemble_top3'][0], names)
        self.assertEqual(by_name['ensemble_top2_flip'][1], ((1.,False),(1.,True)))
        for invalid in [[], ['R07','R07'], ['R01','R02','R03','R04']]:
            with self.assertRaises(ValueError):
                build_inference_jobs(invalid)

    def test_full_cache_rejects_missing_duplicate_images_lost_gt_or_metric_drift(self):
        samples = [dict(im_name=f'image_{i}.jpg',target_cls=[0]*(4 if i<859 else 3)) for i in range(2500)]
        scores = dict(mAP50=.8,mAP50_95=.6,precision=.9,recall=.7)
        require_full_cache_parity(samples,scores,dict(scores))
        broken = list(samples)
        broken[-1] = dict(samples[0])
        for records in [samples[:-1],broken,[dict(row,target_cls=[]) for row in samples]]:
            with self.assertRaises(ValueError):
                require_full_cache_parity(records,scores,dict(scores))
        for changed in [dict(scores,mAP50=.8001),dict(scores,mAP50=float('nan'))]:
            with self.assertRaises(ValueError):
                require_full_cache_parity(samples,scores,changed)
        detailed = dict(scores,per_class={'person':dict(AP50=.9,AP50_95=.7),
            'table':dict(AP50=.7,AP50_95=.5)})
        same_mean_wrong_classes = dict(scores,per_class={'person':dict(AP50=.7,AP50_95=.5),
            'table':dict(AP50=.9,AP50_95=.7)})
        with self.assertRaises(ValueError):
            require_full_cache_parity(samples,detailed,same_mean_wrong_classes)


if __name__=='__main__':
    unittest.main()
