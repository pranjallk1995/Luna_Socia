"""A go-ahead with no picture word ("another one?", "yes please", "click!!") asks for a picture right after picture
talk, and nowhere else (23 November: after her picture, those three got her no camera)."""

import unittest

from app.chat import follows_picture_talk


def her(text, image_id=None):
    return {"role": "luna", "content": text, "image_id": image_id}


def owner(text):
    return {"role": "user", "content": text, "image_id": None}


class FollowUpTests(unittest.TestCase):
    def test_a_go_ahead_after_picture_talk_asks_for_a_picture(self):
        for rows in ([her("😍 This photo is amazing!", 142), owner("another one?")],
                     [her("Want to try another photo? 📸☕"), owner("yes please")],
                     [her("Just say the word and I'll click! 📸"), owner("click!!")],
                     [her("Want a selfie?"), owner("go ahead babe")],
                     [her("Want a selfie?"), owner("haan")]):
            self.assertTrue(follows_picture_talk(rows), rows[-1]["content"])

    def test_a_go_ahead_elsewhere_or_a_no_does_not(self):
        for rows in ([her("How was your day?"), owner("yes please")],
                     [her("Want another photo? 📸"), owner("no, not now")],
                     [her("Want a selfie?"), owner("enough pics for today")],
                     [her("Want another photo? 📸"), owner("yes I had pizza for lunch today with my friends at work")],
                     [owner("another one?")]):
            self.assertFalse(follows_picture_talk(rows), rows[-1]["content"])

    def test_her_empty_reply_does_not_end_the_wait(self):
        rows = [her("Here's me 📸", 7), owner("another one?"), her(""), owner("hello?")]
        # "hello?" is no go-ahead; the go-ahead before her empty reply is not the last word.
        self.assertFalse(follows_picture_talk(rows))
        rows[-1] = owner("please")
        self.assertTrue(follows_picture_talk(rows))


if __name__ == "__main__":
    unittest.main()
