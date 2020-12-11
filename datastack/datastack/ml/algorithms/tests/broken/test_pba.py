import sys
import unittest
import numpy as np

#sys.path.append("..")
import datastack.ml.algorithms.result as result



desc_data = "./data/description.csv"
avro_data = "./data/test.avro"
csv_data = "./data/data.csv"
csv_out_data = "./data/testoutput.csv"
hli_formats = "../../../hli-formats"

class TestResult(unittest.TestCase):

	def setUp(self):
		self._rslt = result.AssociationResult(hli_formats)

	def test_rw_csv(self):
		self.assertTrue(self._rslt.load_csv(csv_data))
		self.assertTrue(self._rslt.load_description(desc_data))
		self.assertTrue(self._rslt.to_csv(csv_out_data))

	def test_rw_avro(self):
		self.assertTrue(self._rslt.load_avro(avro_data))
		self.assertTrue(self._rslt.to_avro(avro_data))

if __name__ == '__main__':
	unittest.main()		