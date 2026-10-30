# description: Four default NN short tests with portable result mail
# email: true
# mode: shell
# parameters:
#   request_id:
#     type: str
#     default: 20261029_nn_metrics_ab_01
#     desc: Unique one-attempt request ID

from src.proj.util.script import ScriptTool


@ScriptTool('nn_metrics_ab')
def main(request_id: str = '20261029_nn_metrics_ab_01', **kwargs):
    import re
    from src.proj import PATH, Proj
    from src.proj.util.script import AutoRunTask
    from src.res.model.diagnostics.nn_metrics_ab import run_suite, save
    from src.res.rl_experimental.bundle import create_bundle, bundle_attachments
    from src.proj.util.web.task_mail import TaskResultMail
    if not re.fullmatch(r'[A-Za-z0-9_-]+', request_id):
        raise ValueError('Invalid request ID')
    parent = PATH.main/'results/nn_training_review/server_ab'/request_id
    if parent.exists():
        raise FileExistsError(f'Use a new request ID; results already exist: {parent}')
    parent.mkdir(parents=True)
    evidence = parent/'evidence'
    AutoRunTask._instances[-1].email_sender = TaskResultMail(parent/'mail')
    try:
        report = run_suite(evidence, PATH.main, request_id)
        print(report['timing'])
        if report['status'] != 'success':
            raise RuntimeError('One or more short tests failed; inspect attached logs')
    except BaseException:
        import traceback
        evidence.mkdir(exist_ok=True)
        save(evidence/'suite_error.json', {'status': 'failed', 'traceback': traceback.format_exc()})
        raise
    finally:
        if evidence.exists():
            metadata = create_bundle(evidence, parent/'bundle')
            if (evidence/'suite.json').exists():
                Proj.email_attachments.append(evidence/'suite.json')
            for part in bundle_attachments(metadata):
                Proj.email_attachments.append(part)
            print(f'NN A/B evidence: {parent}', flush=True)


if __name__ == '__main__':
    main()
