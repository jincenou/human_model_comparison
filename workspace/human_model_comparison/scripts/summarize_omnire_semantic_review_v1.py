"""Record manual visual judgements without editing exports or metric inputs."""
from collections import Counter
from native_reaudit_omnire_v1 import REPORT,EXPORT,read,save,sha

# These labels were made after opening all 100 source/full/actor/export rows,
# all 100 Background-only layers, and all 15 high-contribution node montages.
# They do not depend on any metric value or method ranking.
CONFIRMED_BG={
 'HN0131':'Background-only 中保留与源图一致的浅色衬衣肩胸和举臂外观，actor-only 上身为大面积透明碎片；完整场景中对应人体连续。',
 'HN0142':'Background-only 中明确保留同一灰衣背包男子的上背、背包和右臂；actor-only 上背存在大块透空。',
 'HN0263':'Background-only 直接可见同一深色上衣白裤男子的头胸和白裤，actor-only 头胸大块缺口。',
 'HN0290':'Background-only 直接可见同一正面深上衣白裤男子的头脸和躯干；actor-only 缺失相应连续外观。',
 'HN0311':'Background-only 可见同一灰衣提包男子的上身和提包，actor-only 上躯干几乎透空。',
 'HN0317':'Background-only 明确保留同一黄上衣紫裤女子的肩胸和双腿，actor-only 肩胸透空；早期逐层专项验证同时支持这一结论。',
 'HN0387':'尽管原生头胸严重破碎，Background-only 仍明确保留目标花裙女子的裸露双腿/鞋，与源图位置姿态一致；并非后方车辆节点的外观。',
 'HN0397':'没有目标 actor 节点；Background-only 直接可见源图中的盘坐/蹲坐人物，包括头身、手和裤腿，足以支持静态源时刻补提。',
 'HN0646':'Background-only 中可见目标红色上衣和白色斜带的上躯干与头身；actor-only 上躯干透空。右后白衣邻人由不同节点保留，不合入。',
 'HN0834':'原生整体仍破碎，但 Background-only 中明确保留目标黄色衣服下胸/腹部及深色裙装；actor-only 这些区域透空。身后其他人不视为目标恢复。',
 'HN0861':'原生整体仍破碎，但 Background-only 中可辨目标浅棕衣裙的中部躯干与腿部，同源图中央人物位置一致；左侧儿童和右侧成人是其他身份。',
}

OTHER_VISUAL={
 'HN0038':'RigidNodes 2 是源图左后方卡车，6 是后方轿车；没有目标黑衣站立人的独立身体片段。',
 'HN0099':'DeformableNodes 1/2/7/20/28 是源图中心灰衣背身目标周围的不同邻人，衣着与投影位置不同；不将邻人补为目标。',
 'HN0141':'RigidNodes 15 为远处车；其余为源图灰衣前景男子后的黄衣、黑衣及其他路人，未发现目标身体片段。',
 'HN0181':'RigidNodes 1/17 为后方车辆；原生头肩空洞不是该车辆节点中的目标人体。',
 'HN0189':'RigidNodes 10/11/13 为车辆与货车；未发现长发黑衣目标的身体外观。',
 'HN0212':'RigidNodes 4 为左侧轿车；DeformableNodes 0 对应车旁另一名相邻/部分遮挡的人，不能合并到前方黑衣男子。',
 'HN0216':'RigidNodes 4/5/13/19 为源图后方车辆，其白色车体对透空裤腿区域有贡献但不是人体。',
 'HN0231':'RigidNodes 5/6 为后方车辆/货车；工人残缺头身不应由车辆纹理替代。',
 'HN0272':'RigidNodes 14 为后方黑车，DeformableNodes 3/26/27/31/35/38 为不同空间位置的路人；目标灰衣绿短裤上身碎裂属于原生缺陷。',
 'HN0345':'RigidNodes 6/10/13/14 为后方车辆，10 为源图显著白色车辆；DeformableNodes 11 为更远路人，没有可确认的目标红衣躯干。',
 'HN0387':'RigidNodes 4 为背后经过的黑色轿车，DeformableNodes 21 是远处另一人；目标腿部补提证据来自 Background，不来自这两个节点。',
 'HN0646':'DeformableNodes 46/50/53 为左后暗衣及右后白衣路人，未包含目标红衣白斜带主体。',
 'HN0651':'RigidNodes 3/11 为后方车辆；目标红短裤男子的弱人形仍属原生，不由车体补出。',
 'HN0834':'DeformableNodes 37 是更小、位置在后的另一人，其衣着和投影与前景黄衣女子不同。',
 'HN0850':'RigidNodes 2 为源图蓝衣行人背后的黑车，车头/车窗不能算目标头肩恢复。',
}

# Visible native residuals can be present without a confirmable missing body part.
AMBIGUOUS_BG_RESIDUAL={'HN0004','HN0038','HN0057','HN0090','HN0099','HN0136',
 'HN0140','HN0144','HN0199','HN0211','HN0291','HN0309','HN0325','HN0497',
 'HN0668','HN0690','HN0812','HN0825','HN0866'}

def main():
    visual=read(REPORT/'visual_review_notes.json');identities={r['id']:r for r in read(REPORT/'other_node_identity_audit.json')['records']}
    rows=[]
    for i,base in enumerate(visual['rows']):
        sid=base['id'];v=read(EXPORT/sid/'validation.json');bg=CONFIRMED_BG.get(sid)
        # Correct an inference made from white-composited views after inspecting
        # the actual Background-only layer, where no person is visible.
        note=base['note'] if sid!='HN0233' else '黑上衣红裤女子的原actor已保留原生可见人体；Background-only 未显示目标。颈肩白色空隙被环境合成填暗不能当作身体恢复，复用原actor。'
        if bg:category='confirmed_target_appearance_omitted_in_background'
        elif sid in AMBIGUOUS_BG_RESIDUAL:category='background_added_without_confirmed_target_recovery'
        elif base['native_person_status']=='weak_or_mixed':category='native_reconstruction_itself_underformed'
        else:category='old_actor_preserves_native_visible_target'
        if not v['actor_node_present']:decision='use_background_candidate_no_actor'
        elif bg:decision='use_actor_plus_background_candidate'
        else:decision='reuse_actor_only'
        identity=identities.get(sid);nodes=identity['nodes'] if identity else []
        row=dict(id=sid,method='omnire',native_person_status=base['native_person_status'],
            visual_note=note,visual_class=category,
            BG_recovery_evidence=dict(status='confirmed_by_native_background_layer' if bg else 'not_established',
                explanation=bg or ('背景层有残影或人体样弱支持，但未确认原actor因此漏掉可识别的目标身体/衣服部位；不把背景残影自动等同恢复。' if sid in AMBIGUOUS_BG_RESIDUAL else 'Background-only 中未见与原actor缺失部位对应的明确目标身体外观；白洞填入环境纹理不算恢复。'),
                source=v['source_reference']['reference_rgb'],actor_only=str(REPORT/sid/'actor_only.png'),
                background_only=str(REPORT/sid/'background_only.png'),without_actor=str(REPORT/sid/'without_actor.png'),
                full_scene_anchor_diagnostic=str(REPORT/sid/'full_scene.png')),
            decision=decision,decision_basis='native layer/identity visual attribution; no metric or rank selection',
            actor_node_present=v['actor_node_present'],native_quality_problem=base['native_person_status']=='weak_or_mixed',
            additional_target_loss_observed_in_candidate=False,
            added_background_is_not_automatically_target=True,
            provenance='Complete original actor with native parameters; Background addition is static source-exposure support, not native human 4D motion',
            review_scope=dict(source_full_old_new_sheet=str(REPORT/'semantic_review_sheets'/f'page_{i//10+1:02d}.png'),
                source_actor_background_new_sheet=str(REPORT/'semantic_background_sheets'/f'page_{i//10+1:02d}.png'),
                crop_size_px=188,all_four_views_and_background_layer_actually_opened=True,detector_used_for_review=False),
            other_node_audit=dict(count=len(nodes),nodes=nodes,
                uuid_category_check='all contribution nodes matched uniquely by immutable native size/visibility to dataset UUID; no same UUID as target' if nodes else 'no positive other-node contribution',
                isolated_node_visual_review_performed=sid in OTHER_VISUAL,
                visual_conclusion=OTHER_VISUAL.get(sid,'Identity metadata and source/full/actor/background images inspected; no independent per-node image audit for this low-contribution case.' if nodes else 'No non-target node support'),
                confirmed_target_parts_in_other_nodes=[],automatic_other_node_merge=False),
            immutable_validation=str(EXPORT/sid/'validation.json'),immutable_validation_sha256=sha(EXPORT/sid/'validation.json'))
        if sid in OTHER_VISUAL:row['other_node_audit']['contact_sheet']=str(REPORT/'semantic_node_timing'/sid/'comparison.png')
        rows.append(row)
    assert len(rows)==100 and len({r['id'] for r in rows})==100
    save(REPORT/'semantic_review.json',dict(status='all_100_visual_and_background_layer_review_complete',
        reviewer='Codex actual image review',count=100,decision_counts=dict(Counter(r['decision'] for r in rows)),
        native_person_status_counts=dict(Counter(r['native_person_status'] for r in rows)),
        visual_class_counts=dict(Counter(r['visual_class'] for r in rows)),
        confirmed_background_recovery_ids=sorted(CONFIRMED_BG),
        other_node_identity_objects=len(identities),other_node_identity_nodes=sum(len(x['nodes']) for x in identities.values()),
        isolated_other_node_visual_ids=sorted(OTHER_VISUAL),
        notes=['All100 original full/old/new rows and all100 Background-only layers actually opened.',
          '15 high other-node contributions (>5 percent) also inspected as isolated node images. Remaining37 have metadata plus full/source visual review; no false claim of isolated viewing.',
          'The supplied all-positive-BG exports remain conservative candidates, not automatically the selected final asset.',
          'Technical validity is separate from semantic quality: weak native humans remain eligible according to existing metric protocol.',
          'Old fullscene only target root was at exact exposure; other roots used anchor. New all-node-exposure auxiliary audit supersedes that diagnostic only.',
          'Visual review at188px ROI scale cannot establish pixel-perfect all-part semantic completeness.',
          'HN0233 early preliminary neck/shoulder restoration inference corrected after Background-only inspection.'],
        fullscene_exposure_auxiliary_summary=str(REPORT/'all_node_exposure_aux/summary.json'),
        original_passed_validations_modified=False,original_exports_modified=False,formal_metric_inputs_changed=False,rows=rows))
    print('written',REPORT/'semantic_review.json',Counter(r['decision'] for r in rows))

if __name__=='__main__':main()
