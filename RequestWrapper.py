
import json
from openai import OpenAI
import urllib
import base64
from typing import Any, Dict
import ast
import difflib
import inspect
import json
import logging
import math
import re
import copy
from datetime import datetime
from functools import wraps
import random

from PIL import Image, ImageOps
import io

from typing import List, get_origin
import copy
import random
import string
import requests
import time
import traceback
import yaml
import logging

DEBUG_PRINT = print

def retry(max_attempts=5, default_return=[]):
    DEBUG_PRINT = print

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    time.sleep(random.uniform(1.0, 2.0))
                    DEBUG_PRINT("=" * 100)
                    DEBUG_PRINT(traceback.format_exc())
                    DEBUG_PRINT("=" * 100)
                    if attempts == 1:
                        DEBUG_PRINT(f"[error when retring] args: {[str(args)]}")
                    else:
                        DEBUG_PRINT(f"[error when retring] args: {str(args)[:500]}")
                    DEBUG_PRINT(f"[error when retring] kwargs: {str(kwargs)[:500]}")
                    DEBUG_PRINT(f"[error when retring] sleeping 1~2s. {attempts}/{max_attempts}...")
            # raise Exception("Max retry attempts reached.")
            return default_return

        return wrapper

    return decorator


def encode_image(
    image_path: str,
    max_file_size: int = 10 * 1024 * 1024,  # 最终 JPEG 体积上限（字节）
    max_dimension: int = 2000,  # 最长边上限
    quality: int = 85,  # 初始 JPEG 质量
    min_quality: int = 60,  # 最低 JPEG 质量
    step_quality: int = 5,  # 每次降质步长
    downscale_step: float = 0.9,  # 若降质到下限仍超大，则每轮再缩放 90%
    timeout: int = 15  # URL 读取超时时间（秒）
) -> str:
    """
    读取本地或远程图片，统一转为 JPEG（image/jpeg）并返回 data URI。
    - 纠正 EXIF 方向
    - 透明通道铺白底
    - 若超出尺寸/体积阈值：先按最长边缩放到 max_dimension，再按质量递减压缩；
      若仍超出，则继续按 downscale_step 比例缩放并重试，直到满足或到达质量下限。
    发生异常时，返回原始 image_path（保持与原逻辑一致）。
    """
    try:
        # 1) 读取原始二进制
        if image_path.startswith(('http://', 'https://')):
            req = urllib.request.Request(
                image_path,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        else:
            with open(image_path, "rb") as f:
                raw = f.read()

        # 2) 打开并纠正 EXIF 方向
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img)

        # 3) 透明通道 -> 白底 RGB（JPEG 不支持透明）
        def to_rgb_no_alpha(pil_img, bg=(255, 255, 255)):
            if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
                bg_img = Image.new("RGB", pil_img.size, bg)
                alpha = pil_img.getchannel("A") if "A" in pil_img.getbands() else None
                bg_img.paste(pil_img, mask=alpha)
                return bg_img
            return pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img

        img = to_rgb_no_alpha(img)

        # 4) 若最长边超限，先等比缩放到 max_dimension
        def resize_if_needed(pil_img, limit):
            w, h = pil_img.size
            m = max(w, h)
            if m <= limit:
                return pil_img
            r = limit / float(m)
            new_size = (max(1, int(w * r)), max(1, int(h * r)))
            return pil_img.resize(new_size, Image.Resampling.LANCZOS)

        img = resize_if_needed(img, max_dimension)

        # 5) 保存为 JPEG 的封装
        def save_jpeg_to_bytes(pil_img, q):
            buf = io.BytesIO()
            save_kwargs = dict(format="JPEG", quality=q, optimize=True, progressive=True)
            # Pillow >=9.1 支持 subsampling
            try:
                save_kwargs["subsampling"] = "4:2:0"
            except Exception:
                pass
            pil_img.save(buf, **save_kwargs)
            return buf.getvalue()

        # 6) 质量递减 + 必要时继续缩放，直到满足体积上限或触达下限
        cur_img = img
        cur_quality = quality
        jpeg_bytes = save_jpeg_to_bytes(cur_img, cur_quality)

        # 先尝试在质量下调范围内满足体积
        while len(jpeg_bytes) > max_file_size and cur_quality > min_quality:
            cur_quality = max(min_quality, cur_quality - step_quality)
            jpeg_bytes = save_jpeg_to_bytes(cur_img, cur_quality)

        # 若仍超大，继续按比例缩小图像尺寸并重试（质量维持当前值）
        while len(jpeg_bytes) > max_file_size:
            w, h = cur_img.size
            new_size = (max(1, int(w * downscale_step)), max(1, int(h * downscale_step)))
            if new_size == cur_img.size:  # 已无法再缩
                break
            cur_img = cur_img.resize(new_size, Image.Resampling.LANCZOS)
            jpeg_bytes = save_jpeg_to_bytes(cur_img, cur_quality)

        # 7) Base64 & data URI（固定 image/jpeg）
        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    except Exception as e:
        print(f"encode_image error: {e}")
        # 保持与你原逻辑一致：异常时返回原始路径
        return image_path

def post_request_wrapper(url, data, **kwargs):
    # kwargs is a dict
    default_kwargs = {'headers': None, "timeout": 600, "proxies": None}
    default_kwargs.update(kwargs)

    # print("post_request_wrapper:", generate_curl_command(url, data))
    time.sleep(random.uniform(0, 0.1))  # try to make qps stable and uniform
    response = requests.post(url=url, json=data, **default_kwargs)
    result = response.json()
    return result if result is not None else {}

class RequestWrapperOne:
    def __init__(
        self,
        url: str,
        model_id: str,
        sk: str = None,
        request_method_type: str = None,
        customized_args=None,
        **post_request_kwargs
    ):
        self.url = url
        self.model_id = model_id
        self.sk = sk

        self.request_method_type = self.parse_method_type(request_method_type)
        # DEBUG_PRINT(f"request_method_type: {self.request_method_type}.")
        self.request_method_func = {
            "eb": self._eb_sandbox_request,
            "x1": self._x1_request,
            "vllm": self._vllm_request,
            "api": self._api_request,
            "vllm_with_logprob": self._vllm_with_logprob_request,
        }[self.request_method_type]

        self.customized_args = {}
        self.customized_args.update(customized_args if customized_args is not None else {})

        # keys: headers, timeout, proxies
        self.post_request_kwargs = post_request_kwargs
        headers = {"Content-Type": "application/json"}
        if self.request_method_type == "api":
            headers.update({"Authorization": "Bearer " + self.sk})
        self.post_request_kwargs.update({"headers": headers})

    def parse_method_type(self, request_method_type=None):
        """
        Parse the request method type based on the URL and API key.
        """
        if request_method_type:
            assert request_method_type in ['eb', 'x1', 'api', 'vllm', 'dataeng', 'vllm_with_logprob']
            # TODO: 'vllm_embedding'
            return request_method_type

        _url = self.url if isinstance(self.url, str) else self.url[0]
        _sk = self.sk
        if 'baidu-int.com' in _url and 'eb' in _url:
            return "eb"

        if _sk and _sk.startswith('sk-') and len(_sk) > 40:
            return "api"

        return "vllm"

    def _eb_sandbox_request(self, query_list, history: List = None, media=None, system=None):
        assert self.request_method_type == "eb"
        if media:
            raise NotImplementedError
        if history:
            for h in history:
                assert isinstance(h, list) and len(h) == 2
        else:
            history = []
        if isinstance(query_list, str):
            query_list = [query_list]
        assert isinstance(query_list, list), f"query_list should be a list or a string."

        @retry(max_attempts=3, default_return=("", {}))
        def _request(
            query: str,
            random_session_id: str,
            history: List,
            media=None,
            system: str = None,
            userId=None,
            topp=None,
            temperature=None,
            max_output_tokens=None,
            penalty_score=None
        ):
            assert isinstance(query, str), f"query should be a string, instead of {type(query)}: {query}."
            payload = {
                "text": query,
                "eb_version": "main",
                "model_id": self.model_id,
                "eda_version": "main",
                "session_id": random_session_id,
                "history": history,
                "userId": userId,
                "temperature": temperature,
                "topp": topp,
                "max_output_tokens": max_output_tokens,
                "penalty_score": penalty_score
            }
            if system:
                payload["system"] = system

            for k in list(payload):
                if payload[k] is None:
                    del payload[k]

            result = post_request_wrapper(self.url, payload, **self.post_request_kwargs)
            response_data = result.get("data", {})
            assert response_data is not None, f"result: {result}."
            response = response_data.get("result", None)
            bad_responses = ['正在生成中...']
            if any(bad_ == response for bad_ in bad_responses):
                response = None
            assert response is not None and len(response) > 0, f"response_data: {response_data}."
            return response, result

        random_session_id = ''.join(random.choices(string.ascii_letters + string.digits, k=100))
        responses = []
        raw_results = []
        for _, query in enumerate(query_list):
            response, raw_result = _request(query, random_session_id, history, media, system, **self.customized_args)

            # history for sandbox can be extracted automatically within a same session, identified by session_id.
            responses.append(response)
            raw_results.append(raw_result)
            history = [[query, response]] + history

        self.raw_results = raw_results
        return responses, history

    def _x1_request(self, query_list, history: List = None, media=None, system=None):
        assert self.request_method_type == "x1"

        # url = "http://10.255.72.15:8649/v4/inferencer"
        if history:
            raise NotImplementedError

        if isinstance(query_list, str):
            query_list = [query_list]

        # TODO: check other parameters
        top_p = self.customized_args.get("top_p", 0.95)
        temperature = self.customized_args.get("temperature", 0.6)

        responses = []
        messages = []
        for i, query in enumerate(query_list):
            messages.append({"role": "user", "content": query})
            payload = {
                "config_id": 97,
                "agent_mode": "Chatbot思考模式",
                "think_mode": "Deep",
                "tool_mode": "Sequential",
                "agent_storage": "",
                "sys2_model_id": self.model_id,  # 必填
                "top_p": top_p,  # 必填
                "temperature": temperature,  # 必填
                "session_data": messages,  # 必填
                "global_system_setting": system if system else "",  # 必填，如果有人设信息，可传入；没有就置空
            }
            json_info = requests.post(self.url, json=payload)
            resp = json_info.json()["session_data"][0]["content"]
            thought, response = resp['thoughts'], resp['response']
            responses.append(response)
            messages.append({"role": "sys2", "content": resp})

        return responses, messages

    def _vllm_with_logprob_request(self, query_list, history: list = None, media=None, system=None):
        # precheck
        assert self.request_method_type == "vllm_with_logprob"
        if media:
            raise NotImplementedError
        if history:
            for h in history:
                assert isinstance(h, dict) and "content" in h and \
                       "role" in h and h["role"] in ["user", "assistant", "system"]
        if isinstance(query_list, str):
            query_list = [query_list]
        assert isinstance(query_list, list), f"query_list should be a list or a string."
        n = self.customized_args.get("n", 1)
        if n > 1 and len(query_list) > 1:
            print("[warning] Multi-turn requests will set `n` to 1.")
            n = 1
            self.customized_args["n"] = n
        assert n > 0, "`n` should be set to a positive integer."

        # vllm does not need a _request function thanks to OpenAI API.

        # request
        from openai import OpenAI
        client = OpenAI(
            base_url=self.url,
            api_key=self.sk,
            timeout=600.0
        )

        messages = []
        if system:
            messages = [{"role": "system", "content": system}]

        if (history is None or len(history) == 0):
            pass
        else:
            messages += copy.deepcopy(history)

        responses = []
        logprobs_info = []
        for prompt in query_list:
            messages.append({"role": "user", "content": prompt})

            # n=1, t=0.7, max_tokens=2048, top_p=0.8,
            # extra_body={"repetition_penalty": 1.05, "skip_special_tokens": True}
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                logprobs=True,
                top_logprobs=10,
                **self.customized_args
            )
            if len(completion.choices) > 1:
                responses = [choice.message.content for choice in completion.choices]
                messages.append({"role": "assistant", "content": responses[0]})
                choice = completion.choices[0]
                break
            else:
                response = completion.choices[0].message.content
                messages.append({"role": "assistant", "content": response})
                responses.append(response)
                choice = completion.choices[0]

            if hasattr(choice, "logprobs") and choice.logprobs and hasattr(choice.logprobs, "content"):
                logprob_items = []
                for item in choice.logprobs.content:
                    if item.top_logprobs:
                        prob_dict = {top.token: math.exp(top.logprob) for top in item.top_logprobs}
                        logprob_items.append(prob_dict)
                    else:
                        logprob_items.append({})
                logprobs_info.append(logprob_items)
                messages[-1]["logprobs_info"] = logprob_items

        return responses, messages

    def _vllm_request(self, query_list, history: list = None, media=None, system=None):
        # precheck
        assert self.request_method_type == "vllm"
        if media:
            raise NotImplementedError
        if history:
            for h in history:
                assert isinstance(h, dict) and "content" in h and \
                       "role" in h and h["role"] in ["user", "assistant", "system"]
        if isinstance(query_list, str):
            query_list = [query_list]
        assert isinstance(query_list, list), f"query_list should be a list or a string."
        n = self.customized_args.get("n", 1)
        if n > 1 and len(query_list) > 1:
            print("[warning] Multi-turn requests will set `n` to 1.")
            n = 1
            self.customized_args["n"] = n
        assert n > 0, "`n` should be set to a positive integer."

        # vllm does not need a _request function thanks to OpenAI API.

        # request
        from openai import OpenAI
        client = OpenAI(
            base_url=self.url,
            api_key=self.sk,
            timeout=600.0
        )

        messages = []
        if system:
            messages = [{"role": "system", "content": system}]

        if (history is None or len(history) == 0):
            pass
        else:
            messages += copy.deepcopy(history)

        responses = []
        for prompt in query_list:
            messages.append({"role": "user", "content": prompt})

            # n=1, t=0.7, max_tokens=2048, top_p=0.8, extra_body={"repetition_penalty": 1.05, "skip_special_tokens": True}
            completion = client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **self.customized_args
            )
            if len(completion.choices) > 1:
                responses = [choice.message.content for choice in completion.choices]
                messages.append({"role": "assistant", "content": responses[0]})
                break
            else:
                response = completion.choices[0].message.content
                messages.append({"role": "assistant", "content": response})
                responses.append(response)

        return responses, messages

    def _api_request(self, query_list, history: List = None, media: List[dict] = None, system=None):
        # precheck
        assert self.request_method_type == "api"
        input_img_path = None
        if media:
            # raise NotImplementedError
            input_img_path = media['image']  # 多轮图片
        base64_images = [[]]
        if input_img_path:
            base64_images = [[encode_image(p) for p in paths] for paths in input_img_path]
            # base64_images = [upload_2_public(paths) for paths in input_img_path] # 公网失效

        if history:
            for h in history:
                assert isinstance(h, dict) and "content" in h and \
                       "role" in h and h["role"] in ["user", "assistant", "system"]
        if isinstance(query_list, str):
            query_list = [query_list]
        assert isinstance(query_list, list), f"query_list should be a list or a string."

        def _parse_response_from_result(result):
            if "choices" in result:
                # DEBUG_PRINT(f"[parse info] choices")
                try:
                    choices = result.get("choices", [])
                    assert len(choices) > 0, f"No choices in result: {self.model_id}, {result}."
                    if len(choices) > 1:
                        DEBUG_PRINT(f"get multiple responses: {choices}")
                    response = choices[0]["message"]["content"]
                except:
                    DEBUG_PRINT(f"[parse error] result: {result}")
                    response = result
            elif "text" in result:
                response = result.get('text', None)
            else:
                response = result

            assert isinstance(response, str) and len(response) > 0, f"No response in result: {self.model_id}, {result}."
            return response

        def _request(messages, max_tokens=None):
            payload = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens
            }
            for k in list(payload):
                if payload[k] is None:
                    del payload[k]

            result = post_request_wrapper(self.url, payload, **self.post_request_kwargs)
            response = _parse_response_from_result(result)

            return response, result

        messages = []
        if system:
            messages = [{"role": "system", "content": system}]
        messages += [] if (history is None or len(history) == 0) else copy.deepcopy(history)
        # TODO: check system is given and history contains system too

        responses = []
        raw_results = []
        for query, base64_image in zip(query_list, base64_images):
            text_info = [
                {
                    "type": "text",
                    "text": f"{query}"
                },
            ]
            image_info = []
            if base64_image:
                image_info = [{
                    "type": "image_url",
                    "image_url": {
                        # "url": f"data:image/jpeg;base64,{b64}" if 'http' not in b64 else b64
                        "url": b64
                    }
                } for b64 in base64_image
                ]
            messages += [{"role": "user", "content": text_info + image_info}]

            response, result = _request(messages, **self.customized_args)
            responses.append(response)
            raw_results.append(result)

            messages += [{"role": "assistant", "content": response}]

        self.raw_results = raw_results
        return responses, messages


    @retry(max_attempts=3, default_return=([], []))
    def make_a_valid_request(self, query_list: str or List[str], history: list = None, media: List = None, system=None):
        """
        Make a valid request to the model. with retry 3 times at maximum.
        Args:
            query_list (str or List[str]): The query to send to the model.
            history (List): The history of the conversation so far.
            media (List): The media to send to the model.
            system (str): The system message to send to the model.
        Returns:
            responses (List[str]): The responses from the model.
            messages (List[dict]): The whole messages from the model.
        """
        responses, messages = self.request_method_func(query_list, history, media, system)
        return responses, messages