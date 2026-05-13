"""Prompt templates used by DRIFT.

These strings intentionally mirror the legacy `drift_utils/utils/prompts.py`
templates so the refactor preserves training and inference behavior.
"""

from __future__ import annotations

from drift.utils.constants import COMPRESSION_TOKEN

SYSTEM_PROMPT_AUXILIARY = (
    "Given a context and a question, you need to extract the information "
    "related to this question from the context and compress it into one "
    "compression word (the compressed word is already appended at the end). "
    "If there is no information relevant to the question in the given context, "
    "express a sense of rejection in the compressed wording. After compressing, "
    "please also output your judgment on whether the context is relevant to "
    "the question."
)

USER_PROMPT_AUXILIARY = (
    "The context is: \n<context>{context}</context>\n\n"
    "The question is: \n<question>{question}</question>. "
    f"The compressed word is:{COMPRESSION_TOKEN}."
)

ASSISTANT_PROMPT_AUXILIARY = "{judgement}"

SYSTEM_PROMPT_MAIN = (
    "Please answer the question based on the background. If the background is "
    "irrelevant to the question, answer based on your own knowledge and "
    "disregard the background. Do not say anything else besides answering the "
    "question. If you don't know the answer, just tell me you don't know."
)

USER_PROMPT_MAIN = (
    "The background is: \n<background> {background} </background>\n\n"
    "The question is: \n<question>{question}</question>\n\nYour answer is: "
)

SYSTEM_PROMPT_AUXILIARY_PRETRAIN = (
    "Given a text passage, condense its core concepts into a set of words. "
    "The number of these compressed words is {num}."
    f"The placeholder of compressed word is '{COMPRESSION_TOKEN}'."
)

USER_PROMPT_AUXILIARY_PRETRAIN = (
    "The text you need to condense is: \n<context>{context}</context>\n\n"
    "The compressed words are: "
)

USER_PROMPT_MAIN_PRETRAIN = (
    "Background: <background> {compressed_information} </background>. "
    "Please restate the background information above in your own words to "
    "convey the same meaning: "
)

ASSISTANT_PROMPT_AUXILIARY_PRETRAIN = "{CPS_tokens}"

SYSTEM_PROMPT_AUXILIARY_PRETRAIN_SECOND = (
    "Given several documents and a question, you need to extract the "
    "information from the Documents that is relevant to the Question, and "
    "condense the core concepts of this knowledge into a set of words. \n"
    "Please note that you are only responsible for extracting information "
    "relevant to answering the question. You are not required to reason out "
    "the answer yourself. You are not allowed to fabricate information. You "
    "may only extract and compress relevant information contained in the "
    "documents. Please ensure the completeness and understandability of the "
    "compressed knowledge. \nThe number of these compressed words is {num}."
    f"The placeholder of compressed word is '{COMPRESSION_TOKEN}'."
)

USER_PROMPT_AUXILIARY_PRETRAIN_SECOND = (
    "The documents are: \n<Documents>{document}</Documents>\n\n\n"
    "The question is: \n<Question>{question}</Question>\n\n\n\n"
    "The compressed words of useful information are: "
)

ASSISTANT_PROMPT_AUXILIARY_PRETRAIN_SECOND = "{CPS_tokens}"

USER_PROMPT_MAIN_PRETRAIN_SECOND = (
    "Background: <background> {compressed_information} </background>. "
    "Please restate the background information above in your own words to "
    "convey the same meaning: "
)

USER_PROMPT_MAIN_SFT = (
    "Background: <background> {compressed_information} </background>. "
    "Please answer the following question based on the background. \n"
    "<Question>{question}</Question>\n\n\n\n"
    "Your answer of this question is: "
)

USER_PROMPT_MAIN_MULTI_SFT = (
    "You will be provided with a background consisting of {num} different "
    "paragraphs. Background: <background> {compressed_information} "
    "</background>. Please answer the following question based on the "
    "background. \n<Question>{question}</Question>\n\n\n\n{answer_prefix}"
)

USER_PROMPT_MAIN_KL = (
    "Background:\n{context}\n\nPlease answer the following question based on "
    "the background. \n<Question>{question}</Question>\n\n\n\n"
    "Your answer of this question is: "
)

