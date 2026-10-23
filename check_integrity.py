"""Regression checks: protected data, original baseline and split boundaries."""
import unittest
from evaluate import load_data,metrics
from original.baseline import classify

class IntegrityTests(unittest.TestCase):
    def test_original_data_and_split(self):
        train,test,manifest=load_data()
        self.assertEqual(len(train),72)
        self.assertEqual(len(test),18)
        self.assertEqual(test.groupby('label').size().tolist(),[6,6,6])

    def test_original_baseline_regression(self):
        _,test,_=load_data()
        r=metrics(test.label,[classify(s,t)[0] for s,t in zip(test.project_scope,test.new_request)])
        self.assertAlmostEqual(r['accuracy'],12/18)
        self.assertAlmostEqual(r['clarification_recall'],0.5)
        self.assertEqual(r['confusion_matrix'],[[4,1,1],[0,5,1],[0,3,3]])

if __name__=='__main__':unittest.main()
