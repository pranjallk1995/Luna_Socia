"""A go-ahead with no picture word ("another one?", "yes please", "click!!") asks for a picture right after picture
talk, and nowhere else (23 November: after her picture, those three got her no camera)."""

import unittest

from app.chat import follows_picture_talk

try:
    import test_socia as base
except ImportError:  # run from the repository root
    from tests import test_socia as base


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
                     [her("Want a selfie?"), owner("haan")],
                     [her("Let's capture another moment together! ☕"), owner("ready!")],
                     [her("Sure thing! Let's capture another moment together!"), owner("take one")]):
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


class TakenTests(unittest.TestCase):
    """She had the camera but only wrote "Clicked! Here's your photo!": the picture she meant is taken."""
    setUp = base.ConversationTests.setUp
    tearDown = base.ConversationTests.tearDown
    run_turn = base.ConversationTests.run_turn

    def test_a_photo_she_says_she_clicked_is_taken(self):
        description = "I take a selfie in a cosy cafe, smiling at the camera, warm light on my face."
        events, seen = self.run_turn([[{"content": "📸 Clicked! 💫 Here's your photo! 😍"}]], text="take the photo",
                                     completion=description)
        self.assertIn("picture", seen)
        self.assertIn("image", [event["type"] for event in events])

    def test_picture_talk_unasked_takes_nothing(self):
        events, seen = self.run_turn([[{"content": "I love the photos from our trip! 📸"}]], text="how was your day?",
                                     completion="I take a selfie at the beach.")
        self.assertNotIn("picture", seen)


if __name__ == "__main__":
    unittest.main()
