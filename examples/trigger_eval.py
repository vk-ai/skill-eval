"""Does the skill description fire when it should, and stay quiet on near-misses?"""

from skill_eval import SkillSpec, TriggerCase, run_triggers

skill_v1 = SkillSpec("pdf-forms", text="Use for anything involving PDFs.")
skill_v2 = SkillSpec("pdf-forms", text="Use to fill or read *form fields* in PDF files. Not for converting or summarizing PDFs.")
docx = SkillSpec("docx", text="Edit Word documents.")

cases = [
    TriggerCase("fill", "fill the form fields in invoice.pdf", True),
    TriggerCase("fields", "what fields does tax-form.pdf have?", True),
    TriggerCase("convert", "convert report.docx to PDF", False, note="near-miss: mentions PDF"),
    TriggerCase("summary", "summarize this PDF article", False, note="near-miss: PDF, no form"),
]


def agent(case: TriggerCase, offered, run: int):
    # Stand-in for a real model/CLI that reports which skill it loaded.
    desc = offered[0].text.lower()
    prompt = case.prompt.lower()
    if "docx" in prompt:
        return "docx" if "not for converting" in desc else ["pdf-forms", "docx"]
    if "anything involving pdfs" in desc:
        return "pdf-forms" if "pdf" in prompt else None
    return "pdf-forms" if ("field" in prompt or "form" in prompt) else None


if __name__ == "__main__":
    for skill in (skill_v1, skill_v2):
        print(run_triggers(cases, skill, agent, k=3, competing=[docx]).table())
