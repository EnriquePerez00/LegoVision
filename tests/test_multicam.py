import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from inference.api import api_generate_single_piece_renders, api_inference_multicam_set

class TestMulticamEndpointsDirect(unittest.TestCase):

    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("builtins.open")
    @patch("json.load")
    def test_generate_single_piece_renders_success(self, mock_json_load, mock_open, mock_exists, mock_run):
        # Configure mocks
        mock_exists.side_effect = lambda path: True
        mock_run.return_value = MagicMock(returncode=0)
        mock_json_load.return_value = {
            "set_id": "75078-1",
            "pieces_count": 1,
            "renders": []
        }

        # Call function directly
        data = api_generate_single_piece_renders(set_id="75078-1")
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["metadata"]["set_id"], "75078-1")

    @patch("inference.api.get_classifier")
    @patch("os.path.exists")
    @patch("builtins.open")
    @patch("json.load")
    @patch("PIL.Image.open")
    def test_inference_multicam_set_success(self, mock_image_open, mock_json_load, mock_open, mock_exists, mock_get_classifier):
        # Configure mocks
        mock_exists.side_effect = lambda path: True
        
        # Mock metadata JSON load
        mock_json_load.return_value = {
            "set_id": "75078-1",
            "pieces_count": 1,
            "renders": [
                {
                    "ref": "3001",
                    "name": "Brick 2 x 4",
                    "color_hex": "#FF0000",
                    "color_code": "4",
                    "cameras": {
                        "cenital": {
                            "file_name": "single_3001_4_cenital.png",
                            "bbox_norm": [0.1, 0.1, 0.9, 0.9],
                            "image_url": "/renders/multicam/single_3001_4_cenital.png"
                        },
                        "lateral_l": {
                            "file_name": "single_3001_4_lateral_l.png",
                            "bbox_norm": [0.1, 0.1, 0.9, 0.9],
                            "image_url": "/renders/multicam/single_3001_4_lateral_l.png"
                        },
                        "lateral_r": {
                            "file_name": "single_3001_4_lateral_r.png",
                            "bbox_norm": [0.1, 0.1, 0.9, 0.9],
                            "image_url": "/renders/multicam/single_3001_4_lateral_r.png"
                        }
                    }
                }
            ]
        }
        
        # Mock DINOv2 Classifier
        mock_clf = MagicMock()
        mock_clf.is_ready.return_value = True
        mock_clf.classify.return_value = [
            {"part_ref": "3001", "part_name": "Brick 2 x 4", "score": 0.95, "detected_color": "4"}
        ]
        mock_get_classifier.return_value = mock_clf

        # Mock PIL image size and convert method
        mock_img = MagicMock()
        mock_converted = MagicMock()
        mock_converted.size = (100, 100)
        mock_img.convert.return_value = mock_converted
        mock_image_open.return_value = mock_img

        # Call function directly
        data = api_inference_multicam_set(set_id="75078-1")
        self.assertEqual(data["status"], "success")
        self.assertIn("mean_accuracy", data)
        self.assertIn("worst_3_pieces", data)
        self.assertIn("inventory", data)
        self.assertIn("results", data)
        
        # Verify consensus prediction was correct
        results = data["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["consensus_ref"], "3001")
        self.assertTrue(results[0]["is_consensus_correct"])

if __name__ == "__main__":
    unittest.main()
