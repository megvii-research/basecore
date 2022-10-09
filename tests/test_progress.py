#!/usr/bin/env python3

import unittest

from basecore.engine import Progress


class TestProgress(unittest.TestCase):

    def setUp(self):
        self.p = Progress(max_info=[10, 10])

    def test_total_progress(self):
        count = 0
        while not self.p.reach_epoch_end():
            while not self.p.reach_iter_end():
                count += 1
                self.p.next_iter()
            self.p.next_epoch()

        self.assertEqual(count, 100)

    def test_progress_usage(self):
        self.assertTrue(self.p.is_first_iter())
        gt_string = "epoch:1/10, iter:1/10"
        self.assertEqual(str(self.p), gt_string)
        self.p.next_iter()
        gt_str_list = ["epoch1", "iter2"]
        p_str = self.p.progress_str_list()
        self.assertTrue(all(x == y for x, y in zip(p_str, gt_str_list)))

    def test_state_dict(self):
        self.p.iter = 5
        self.p.epoch = 6
        states = self.p.state_dict()

        new_p = Progress(max_info=[10, 10])
        new_p.load_state_dict(states)
        self.assertEqual(self.p, new_p)

    def test_scale(self):
        e_infos = [1, 4]
        out = self.p.scale_to_iterwise(e_infos)
        self.assertEqual(out, [10, 40])
        out = self.p.scale_to_epochwise(out)
        self.assertEqual(out, e_infos)

    def test_iter_info(self):
        self.p.iter = 4
        self.p.epoch = 6
        self.assertEqual(self.p.current_iter(), 53)
        self.assertEqual(self.p.left_iter(), 100 - 53)


if __name__ == "__main__":
    unittest.main()
