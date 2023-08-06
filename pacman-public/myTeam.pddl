;Header and description

(define (domain pacman_bool)

    ;remove requirements that are not needed
    (:requirements :strips :typing :negative-preconditions)
    (:types 
        enemy team - object
        enemy1 enemy2 - enemy
        ally current_agent - team
    )

    ; un-comment following line if constants are needed
    ;(:constants )
    ;(:types food)
    (:predicates 

        ;Basic predicates
        (enemy_around ?e - enemy ?a - team) ;enemy ?e is within 4 grid distance with agent ?a
        (is_pacman ?x) ; if an agent is pacman 
        (food_in_backpack ?a - team)  ; have food in backpack
        (food_available) ; still have food on enemy land

        ;Predicates for virtual state to set goal states
        (defend_foods) ;The environment do not collect state for this predicates, this is a virtual effect state for action patrol 
        
        ; (go_to_eat_enemy ?a - team ?e - enemy) ; go towards the enemy to eat them

        (time_running_out) ; game time running out

        (patrol_threshold) ; condition defined in pddl state collection to trigger patrolling
        (get_in_there_buddy) ; condition defined in pddl state collection to trigger diving in without a care
        
        ;Advanced predicates
        ;These predicates are currently not used and consider the state of other agent 
        (enemy_long_distance ?e - enemy ?a - current_agent) ; noisy distance return longer than 25 
        (enemy_medium_distance ?e - enemy ?a - current_agent) ; noisy distance return longer than 15 
        (enemy_short_distance ?e - enemy ?a - current_agent) ; noisy distance return shorter than 15 

        (3_food_in_backpack ?a - team) ; more than 3 food in backpack
        (5_food_in_backpack ?a - team)  ; more than 5 food in backpack
        (10_food_in_backpack ?a - team)    ; more than 10 food in backpack
        (15_food_in_backpack ?a - team)    ; more than 10 food in backpack
        (20_food_in_backpack ?a - team)    ; more than 20 food in backpack

        (near_food ?a - current_agent)  ; a food within 4 grid distance 
        (near_capsule ?a - current_agent)   ;a capsule within 4 grid distance

        (capsule_available) ; capsule available on map
        (capsule_available_enemy) ; capsule is available for enemy to take

        (winning)   ; is the team score more than enemy team
        (winning_gt3) ; is the team score 3 more than enemy team
        (winning_gt5)    ; is the team score 5 more than enemy team
        (winning_gt8)    ; is the team score 8 more than enemy team

        (winning_gt10)  ; is the team score 10 more than enemy team
        (winning_gt20)  ; is the team score 20 more than enemy team

        (near_ally  ?a - current_agent) ; is ally near 4 grid distance
        (is_scared ?x) ;is enemy, current agent, or the ally in panic (due to capsule eaten by other side)
        (in_enemy_territory ?a - current_agent) ; is the agent in enemy territory?
        
        (food_disappeared) ; food disappeared

        ;Cooperative predicates
        ;The states of the following predicates are not collected by demo team_ddl code;
        ;To use these predicates, You have to collect the corresponding states when preparing pddl states in the code.
        ;These predicates describe the current action of your ally
        (eat_enemy ?a - ally)
        (go_home ?a - ally)
        (go_enemy_land ?a - ally)
        (eat_capsule ?a - ally)
        (eat_food ?a - ally)
        (on_defence ?a - ally)

    )

    ;define actions here

    (:action attack
        :parameters (?a - current_agent ?a2 - ally ?e1 - enemy1 ?e2 - enemy2 )
        :precondition (and 
            (not (is_pacman ?e1)) 
            (not (is_pacman ?e2)) 
            (food_available)  
            (in_enemy_territory ?a)
            (not (get_in_there_buddy))

            
        )
        :effect (and 
            (not (food_available))

        )
    )

    ; end game greedy state
    (:action dive_bomb
        :parameters (?a - current_agent)
        :precondition (and 
            (get_in_there_buddy)
        )
        :effect (and
            (not (food_available))
        )

    )

    ; action to rush
    (:action rush
        :parameters (?a - current_agent ?a2 - ally ?e1 - enemy1 ?e2 - enemy2 )
        :precondition (and 
            (not (is_pacman ?e1)) 
            (not (is_pacman ?e2)) 
            (not (in_enemy_territory ?a))

            
        )
        :effect (and 
            (is_pacman ?a)
            (in_enemy_territory ?a)
        )
    )


    ; if 3 foods in the backpack, must eat fast
    (:action consume_food_quicc
        :parameters(?a - current_agent)
        :precondition (and 
            (15_food_in_backpack ?a)
            (is_pacman ?a)
            ; (food_available)
        )
        :effect (and 
            (not (is_pacman ?a))
            (not (15_food_in_backpack ?a))
            ; (food_available)
        )
    )

    ; if invader spotted this move signals active defence
    (:action defence_active
        :parameters (?a - current_agent ?a2 - ally ?e - enemy)
        :precondition (and 
            (is_pacman ?e)
            (not (is_pacman ?a))
            ; (not (food_disappeared))
            (enemy_around ?e ?a)
            ; (go_enemy_land ?a2)
            
        )
        :effect (and 
            (not (is_pacman ?e))
            (not (enemy_around ?e ?a))
            ; (not (food_disappeared))
            ; (on_defence ?a)
        )
    )

    ; action to define passive defence
    (:action defence_passive
        :parameters(?a - current_agent ?a2 ally ?e - enemy)
        :precondition (and 
            (is_pacman ?e)
            (not (is_pacman ?a))
            (not (enemy_around ?e ?a))
            (not (food_disappeared))
        )
        :effect (and 
            (not (is_pacman ?e))
            (enemy_around ?e ?a)
        )


     )

    ; action to define food disappearing and pinging the location
    (:action defence_ping
        :parameters (?a - current_agent ?a2 - ally ?e - enemy)
        :precondition (and 
            (is_pacman ?e)
            (not (is_pacman ?a))
            (food_disappeared)
            ; (not (enemy_around ?e ?a))
            ; (go_enemy_land ?a2)
            
        )
        :effect (and 
            (not (is_pacman ?e))
            (not (food_disappeared))
            ; (enemy_around ?e ?a)
            ; (on_defence ?a)
        )
    )

    ; action to define keeping distance but staying within sight of the invader
    (:action defence_avoid
        :parameters (?a - current_agent ?a2 - ally ?e - enemy)
        :precondition (and 
            (is_pacman ?e)
            (not (is_pacman ?a))
            (is_scared ?a)
            (enemy_around ?e ?a)

            
        )
        :effect (and 
            (not (is_pacman ?e))
            (not (is_scared ?a))
            ; (not (food_disappeared))
            ; (on_defence ?a)
        )
    )

    ; when no more food left, return home
    (:action return_home
        
        :parameters(?a - current_agent)
        :precondition (and 
            (not (food_available))
        )
        :effect (and 
            (defend_foods)
            
        )
    )


    ; action to go home
    (:action go_home
        :parameters (?a - current_agent)
        :precondition (
            and (is_pacman ?a) 
        )
        :effect (and 
            (not (is_pacman ?a))
        )
    )

    ; action to patrol the border when patrol threshold called
    (:action patrol
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and 
            (not (is_pacman ?a))
            (not (is_pacman ?e1))
            (not (is_pacman ?e2))
            (patrol_threshold)
        )
        :effect (and 
            (defend_foods)
        )
    )

)